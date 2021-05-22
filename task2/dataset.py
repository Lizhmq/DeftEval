import os
import pickle
import random
import json

import numpy as np
from numpy.core.shape_base import block
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils import process_bio


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_name='train.pkl', block_size=512):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_name[:-4] +"_blocksize_%d" %
                                   (block_size)+"_wordsize_%d" % (world_size)+"_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels = datas["x"], datas["y"]

        else:
            self.inputs = []
            self.labels = []
            datafile = os.path.join(args.data_dir, file_name)
            datas = pickle.load(open(datafile, "rb"))
            inputs, labels  = datas["x"], datas["y"]

            length = len(inputs)

            for idx, (data, label) in enumerate(zip(inputs, labels)):
                if idx % world_size == local_rank:
                    code = data
                    code_tokens = tokenizer.tokenize(code)[:block_size-2]
                    code_tokens = [tokenizer.cls_token] + \
                        code_tokens + [tokenizer.sep_token]
                    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
                    padding_length = block_size - len(code_ids)
                    code_ids += [tokenizer.pad_token_id] * padding_length
                    self.inputs.append(code_ids)
                    self.labels.append(label)

                # if idx % (length // 10) == 0:
                #     percent = idx / (length//10) * 10
                #     logger.warning("Rank %d, load %d" % (local_rank, percent))

            with open(cached_file, 'wb') as handle:
                pickle.dump({"x": self.inputs, "y": self.labels},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item])


class JointDataset(Dataset):        # jointly training classification and sequence labeling
    def __init__(self, tokenizer, args, logger, file_name, label_map=None, block_size=512, aug=False):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_name+"_blocksize_%d" %
                                   (block_size)+"_wordsize_%d" % (world_size)+"_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            if file_name == 'all_train.pkl':
                logger.warning(
                    "Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.xs, self.ys, self.bios = datas["x"], datas["y"], datas["bio"]
                self.x_mask, self.bio_mask = datas["xm"], datas["bm"]
                self.label_map = datas["label_map"]
        else:
            self.xs, self.ys, self.bios = [], [], []
            self.x_mask, self.bio_mask = [], []
            datafile = os.path.join(args.data_dir, file_name)
            if file_name == 'all_train.pkl':
                logger.warning(
                    "Creating features from dataset file at %s", datafile)
            datas = pickle.load(open(datafile, "rb"))
            xs, ys, bios = datas["x"], datas["y"], datas["bio"]
            xs = [sent.split(" ") for sent in xs]
            assert(len(xs) == len(ys))
            assert(len(ys) == len(bios))
            for i in range(len(xs)):
                assert(len(xs[i]) == len(bios[i]))
            
            tmpset = set()
            for bio in bios:
                for b in bio:
                    tmpset.add(b)
            length = len(xs)

            label_list = list(tmpset)
            if label_map == None:
                self.label_map = {label : i for i, label in enumerate(label_list, 0)}
            else:
                self.label_map = label_map

            brdid = self.label_map["B-Referential-Definition"]
            bqid = self.label_map["B-Qualifier"]
            brtid = self.label_map["B-Referential-Term"]

            for exidx, (x, y, bio) in enumerate(zip(xs, ys, bios)):
                if exidx % world_size == local_rank:

                    tokens = x
                    assert len(tokens) == len(bio)
                    subtokens = tokenizer.tokenize(" ".join(tokens))
                    input_ids = tokenizer.convert_tokens_to_ids(subtokens)
                    input_mask, labels, label_mask = process_bio(tokens, subtokens, bio)
                    label_ids = list(map(lambda t: self.label_map[t] if t in self.label_map else -1, labels))
                    
                    input_ids = input_ids[:block_size-2]
                    input_mask = input_mask[:block_size-2]
                    label_ids = label_ids[:block_size-2]
                    label_mask = label_mask[:block_size-2]

                    subtokens.insert(0, tokenizer.bos_token)
                    input_ids.insert(0, tokenizer.bos_token_id)
                    input_mask.insert(0, 1)
                    label_ids.insert(0, -1)
                    label_mask.insert(0, 0)

                    subtokens.append(tokenizer.sep_token)
                    input_ids.append(tokenizer.sep_token_id)
                    input_mask.append(1)
                    label_ids.append(-1)
                    label_mask.append(0)

                    while len(input_ids) < block_size:
                        input_ids.append(tokenizer.pad_token_id)
                        input_mask.append(0)
                        label_ids.append(-1)
                        label_mask.append(0)

                    assert len(input_ids) == block_size
                    assert len(input_mask) == block_size
                    assert len(label_ids) == block_size
                    assert len(label_mask) == block_size
                    dup = 1
                    if file_name != "test2.pkl":
                        if brtid in label_ids:
                            dup = 5
                        elif bqid in label_ids:
                            dup = 3
                        if brdid in label_ids:
                            dup = 2
                    for _ in range(dup):
                        self.xs.append(input_ids)
                        self.x_mask.append(input_mask)
                        self.ys.append(y)
                        self.bios.append(label_ids)
                        self.bio_mask.append(label_mask)
                    if exidx / world_size < 5:
                        logger.info("*** Example ***")
                        logger.info("tokens: %s" % " ".join(
                                [str(x) for x in subtokens]))
                        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                        # logger.info("label: %s (id = %d)" % (example.label, label_ids))

                if exidx % (length // 10) == 0:
                    percent = exidx / (length // 10) * 10
                    logger.warning("Rank %d, load %d" % (local_rank, percent))

            if file_name == 'all_train.pkl':
                logger.warning("Rank %d Training %d samples" %
                               (local_rank, len(self.xs)))
                logger.warning(
                    "Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump({"x": self.xs, "y": self.ys, "bio": self.bios,
                             "xm": self.x_mask, "bm": self.bio_mask, "label_map": self.label_map},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return torch.tensor(self.xs[item]), torch.tensor(self.ys[item]), torch.tensor(self.bios[item]), \
                torch.tensor(self.x_mask[item]), torch.tensor(self.bio_mask[item])
