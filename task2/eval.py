from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AdamW, BertConfig,
                                  RobertaForTokenClassification, RobertaTokenizer, get_linear_schedule_with_warmup)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from dataset import ClassifierDataset, JointDataset
from model import JointModel, build_model


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def gen_2(args, model, tokenizer, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=False)
    num_sample = 0
    model.eval()
    preds = []
    ii = 0
    for batch in tqdm(eval_dataloader):
        # ii += 1
        # if ii == 21:
        #     break
        h = [batch]
        for i in range(len(h[0])):
            h[0][i] = h[0][i].to(args.device)
        x, y, bio, x_mask, bio_mask = h[0]

        with torch.no_grad():
            pred = model.forward2(x, y, x_mask, bio_mask).cpu().numpy()
            nums = torch.sum(bio_mask, dim=1)
            batch_pred = []
            start = 0
            for num in nums:
                batch_pred.append(pred[start:start+num])
                start += num
            preds.extend(batch_pred)
            num_sample += y.size()[0]
    return preds

def inv_map(preds, dic, output_file):
    inv_dic = dict()
    for key, value in dic.items():
        inv_dic[value] = key
    labels = []
    for pred in preds:
        labels.append([inv_dic[v] for v in pred])
    with open(output_file, "a") as f:
        f.writelines([" ".join(line) + "\n" for line in labels])
    return labels



def evaluate_1(args, model, tokenizer, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=False)
    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    num_sample = 0
    model.eval()
    preds, golds = np.array([]), np.array([])
    for batch in tqdm(eval_dataloader):
        inputs, labels = batch
        batch_max_length = inputs.ne(
            tokenizer.pad_token_id).sum(-1).max().item()
        inputs = inputs[:, :batch_max_length]
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            pred, gold, loss = model.forward1(inputs, input_mask=inputs.ne(
                tokenizer.pad_token_id), y=labels)
            preds = np.append(preds, pred)
            golds = np.append(golds, gold)
            
        num_sample += labels.size()[0]

    target_names = ["class_0", "class_1"]
    print(classification_report(golds, preds, target_names=target_names, digits=3))
    return 



def valid_loss(args, model, tokenizer, eval_dataset, prefix="", eval_when_training=False):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    losses = []
    model.eval()

    for batch in eval_dataloader:
        x, y, bio, x_mask, bio_mask = batch
        x = x.to(args.device)
        y = y.to(args.device)
        bio = bio.to(args.device)
        x_mask = x_mask.to(args.device)
        bio_mask = bio_mask.to(args.device)

        with torch.no_grad():
            output = model(x, y, x_mask, bio, bio_mask)
            losses.append(output.item())
    result = {
        "cross entropy": sum(losses) / len(losses)
    }

    output_eval_file = os.path.join(
        eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        #logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            #logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result





def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    ## Other parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data path.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The dir of pretrained model.")
    parser.add_argument("--block_size", default=384, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--lambd', type=float, default=1,
                        help="hyper parameter for joint learning")

    args = parser.parse_args()
    args.local_rank = -1
    args.device = torch.device("cuda", 0)

    label_list = ["O", "B-Term", "I-Term", "B-Definition", \
                "I-Definition", "B-Alias-Term", "I-Alias-Term", \
                "B-Referential-Definition", "I-Referential-Definition", \
                "B-Referential-Term", "I-Referential-Term", "B-Qualifier", "I-Qualifier"]
    label_map = dict()
    for i, lab in enumerate(label_list):
        label_map[lab] = i
    
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_dir)
    train_dataset = JointDataset(tokenizer, args, logger, file_name="all_train.pkl", label_map=label_map, block_size=512)
    valid_dataset = JointDataset(tokenizer, args, logger, file_name="all_dev.pkl", label_map=label_map, block_size=512)
    test_dataset = JointDataset(tokenizer, args, logger, file_name="test2.pkl", label_map=label_map, block_size=512)
    # test_dataset = ClassifierDataset(tokenizer, args, logger, file_name="test1.pkl", block_size=512)
    bio_size = len(train_dataset.label_map)
    model = build_model(args, bio_size, args.pretrain_dir)
    model.tasks = "2"
    # results = valid_loss(args, model, tokenizer, eval_dataset=valid_dataset, eval_when_training=True)
    # print(results)
    # evaluate_1(args, model, tokenizer, eval_dataset=test_dataset)
    dic = train_dataset.label_map
    
    preds = gen_2(args, model, tokenizer, test_dataset)
    # results = valid_loss(args, model, tokenizer, eval_dataset=test_dataset, eval_when_training=True)
    # print(results)
    inv_map(preds, dic, "test-2000.txt")

if __name__ == "__main__":
    main()