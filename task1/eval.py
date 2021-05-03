import os, logging
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import ClassifierDataset
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, AdamW,  get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = ClassifierDataset(
        tokenizer, args, logger, file_type='test' if evaluate else 'train', block_size=args.block_size)
    return dataset


def evaluate(args, model, tokenizer):
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=False)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    correct = 0.0
    correct_y = []
    predict_y = []
    num_sample = 0
    model.eval()

    for batch in tqdm(eval_dataloader):
        inputs, labels = batch
        batch_max_length = inputs.ne(
            tokenizer.pad_token_id).sum(-1).max().item()
        inputs = inputs[:, :batch_max_length]
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask=inputs.ne(
                tokenizer.pad_token_id))
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            correct += preds.eq(labels).sum()
            correct_y.extend(labels.cpu().numpy())
            predict_y.extend(preds.cpu().numpy())

        num_sample += labels.size()[0]

    acc = correct / num_sample
    result = {
        "accuracy": float(acc)
    }
    target_names = ["class_0", "class_1"]
    print(classification_report(correct_y, predict_y, target_names=target_names))

    return result


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    ## Other parameters
    parser.add_argument("--evaluate_dir", default="", type=str,
                        help="The dir of pretrained model.")
    parser.add_argument("--block_size", default=384, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.local_rank = -1
    args.output_dir = args.data_dir
    args.device = torch.device("cuda")

    tokenizer = RobertaTokenizer.from_pretrained(args.evaluate_dir)
    model = RobertaForSequenceClassification.from_pretrained(args.evaluate_dir).to(args.device)
    evaluate(args, model, tokenizer)



if __name__ == "__main__":
    main()