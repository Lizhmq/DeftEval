import os
import pickle
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter



class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=384):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d" %
                                   (block_size)+"_wordsize_%d" % (world_size)+"_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            if file_type == 'train':
                logger.warning(
                    "Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels = datas["inputs"], datas["labels"]

        else:
            self.inputs = []
            self.labels = []
            if file_type != "test":
                datafile = os.path.join(args.data_dir, "train.pkl")
                datafile2 = os.path.join(args.data_dir, "dev.pkl")
            else:
                # datafile = os.path.join(args.data_dir, "test.pkl")
                datafile = os.path.join(args.data_dir, "dev.pkl")
            if file_type == 'train':
                logger.warning(
                    "Creating features from dataset file at %s", datafile)
            datas = pickle.load(open(datafile, "rb"))
            datas2 = pickle.load(open(datafile2, "rb"))
            inputs, labels  = datas
            inputs2, labels2 = datas2

            if file_type == "train":
                inputs, labels = inputs, labels
            elif file_type == "dev":
                inputs, labels = inputs2, labels2
            length = len(inputs)

            for idx, (data, label) in enumerate(zip(inputs, labels)):
                if idx % world_size == local_rank:
                    code = " ".join(data)
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

            if file_type == 'train':
                logger.warning("Rank %d Training %d samples" %
                               (local_rank, len(self.inputs)))
                logger.warning(
                    "Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump({"inputs": self.inputs, "labels": self.labels},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item])
