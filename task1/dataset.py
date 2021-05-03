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
            datafile = os.path.join(args.data_dir, "train.pkl")
            datafile2 = os.path.join(args.data_dir, "dev.pkl")
            datafile3 = os.path.join(args.data_dir, "test.pkl")
            # datafile = os.path.join(args.data_dir, "dev.pkl")
            if file_type == 'train':
                logger.warning(
                    "Creating features from dataset file at %s", datafile)
            datas = pickle.load(open(datafile, "rb"))
            datas2 = pickle.load(open(datafile2, "rb"))
            datas3 = pickle.load(open(datafile3, "rb"))
            inputs, labels  = datas
            inputs2, labels2 = datas2
            inputs3, labels3 = datas3

            if file_type == "train":
                inputs, labels = inputs, labels
            elif file_type == "dev":
                inputs, labels = inputs2, labels2
            elif file_type == "test":
                inputs, labels = inputs3, labels3
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


class Vocabulary:

    PAD = 0
    UNK = 1

    def __init__(self, vocab_size):
        self.word2count = {}
        self.word2idx = None
        self.idx2word = ["<pad>", "<unk>"]
        self.num_words = 2
        self.num_sentences = 0
        self.vocab_size = vocab_size

    def fit_training_corpus(self, sents):
        for sent in sents:
            self.add_sentence(sent)
        sorted_dict = sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size-2]
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for word, cnt in sorted_dict:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)
        self.num_sentences += 1

    def to_word(self, index):
        if index < self.vocab_size:
            return self.idx2word[index]
        else:
            return "<unk>"

    def to_idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx["<unk>"]

    def seq2idxs(self, seq, cut_and_pad, max_len=None):
        if cut_and_pad:
            assert(max_len is not None)
        ret = [self.to_idx(token) for token in seq]
        if cut_and_pad:
            ret = ret[:max_len]
            ret.extend([self.PAD] * (max_len - len(ret)))
        return ret


class LSTMDataset(Dataset):
    
    def __init__(self, file_path, output_dir, file_type='train', vocab_size=16000, max_len=384):
        
        file_type = file_type.lower()
        assert file_type in ['train', 'test', 'dev']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cached_file = os.path.join(output_dir, file_type+"_maxlen_%d" % (max_len))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels = datas["inputs"], datas["labels"]
        else:
            datafile = os.path.join(file_path, "train.pkl")
            datafile2 = os.path.join(file_path, "dev.pkl")
            datafile3 = os.path.join(file_path, "test.pkl")

            datas = pickle.load(open(datafile, "rb"))
            datas2 = pickle.load(open(datafile2, "rb"))
            datas3 = pickle.load(open(datafile3, "rb"))
            inputs, labels  = datas
            inputs2, labels2 = datas2
            inputs3, labels3 = datas3


            self.vocab_size = vocab_size
            self.vocab = Vocabulary(vocab_size)
            self.vocab.fit_training_corpus(inputs)
            self.idx2word = self.vocab.idx2word
            self.word2idx = self.vocab.word2idx

            self.inputs, self.labels = inputs, labels
            if file_type == "dev":
                self.inputs, self.labels = inputs2, labels2
            elif file_type == "test":
                self.inputs, self.labels = inputs3, labels3
            self.inputs = torch.LongTensor([self.vocab.seq2idxs(seq, cut_and_pad=True, max_len=max_len) for seq in self.inputs])
            self.labels = torch.LongTensor(self.labels)
            with open(cached_file, 'wb') as handle:
                pickle.dump({"inputs": self.inputs, "labels": self.labels},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        
        return self.inputs[idx], self.labels[idx]
    
    def __len__(self):
        
        return len(self.inputs)