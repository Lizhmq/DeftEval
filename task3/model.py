import argparse
import csv
import enum
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel

import pickle
from torchcrf import CRF




class JointModel(nn.Module):

    def __init__(self, hidden, classes, bio_classes, lambd, args, device):
        super().__init__()
        self.args = args
        self.lambd = lambd
        self.device = device
        self.classes = classes
        self.bio_classes = bio_classes
        self.dropout = nn.Dropout(0.1)
        self.gen_classifier = nn.Linear(hidden, classes)
        self.bio_classifier = nn.Linear(hidden, bio_classes)
        self.bert = RobertaModel.from_pretrained(self.args.pretrain_dir)
        self._init_cls_weight()
        self.crf = CRF(self.bio_classes, batch_first=True)


    def _init_cls_weight(self, initializer_range=0.02):
        for layer in (self.gen_classifier, self.bio_classifier):
            layer.weight.data.normal_(mean=0.0, std=initializer_range)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
        torch.save(self.gen_classifier.state_dict(), os.path.join(path, "gen_cls.bin"))
        torch.save(self.bio_classifier.state_dict(), os.path.join(path, "bio_cls.bin"))
        torch.save(self.crf.state_dict(), os.path.join(path, "crf.bin"))
        
    def from_pretrained(self, path):
        self.bert = RobertaModel.from_pretrained(path)
        self.gen_classifier.load_state_dict(torch.load(os.path.join(path, "gen_cls.bin"), map_location=self.device))
        self.bio_classifier.load_state_dict(torch.load(os.path.join(path, "bio_cls.bin"), map_location=self.device))
        self.crf.load_state_dict(torch.load(os.path.join(path, "crf.bin"), map_location=self.device))
        return self

    def forward2(self, input_ids, input_mask, out_mask):
        out_mask = (out_mask == 1)
        sequence_output = self.bert(input_ids, input_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = sequence_output.masked_select(out_mask.unsqueeze(-1))
        valid_output = valid_output.view(-1, feat_dim)
        output = torch.mean(valid_output, dim=0)
        return output



def build_model(args, bio_size, load_path=None):
    model = JointModel(768, 2, bio_size, args.lambd, args, args.device)
    if load_path is not None:
        model = model.from_pretrained(load_path).to(args.device)
    return model


def process_bpe(tokens, subtokens, out_mask, bpe_indicator='Ġ'):
    starts = [0]
    starts += list(filter(lambda j: subtokens[j][0] == bpe_indicator, range(len(subtokens))))
    assert len(starts) == len(tokens)
    assert len(tokens) == len(out_mask)
    input_mask = [1] * len(subtokens)
    gen_mask = [0] * len(subtokens)
    for i, st in enumerate(starts):
        if out_mask[i] == 0:
            continue
        if i == len(starts) - 1:
            end = len(starts)
        else:
            end = starts[i + 1]
        for idx in range(st, end):
            gen_mask[idx] = 1
    return input_mask, gen_mask


def gen_rep(model, tokenizer, x, out_mask, block_size=512):
    assert len(x) == len(out_mask)
    subtokens = tokenizer.tokenize(" ".join(x))
    input_mask, out_mask = process_bpe(x, subtokens, out_mask)
    
    subtokens = subtokens[:block_size-2]
    input_mask = input_mask[:block_size-2]
    out_mask = out_mask[:block_size-2]
    
    subtokens.insert(0, tokenizer.bos_token)
    subtokens.append(tokenizer.sep_token)
    input_mask.insert(0, 1)
    input_mask.append(1)
    out_mask.insert(0, 0)
    out_mask.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(subtokens)
    input_ids = torch.tensor([input_ids], device=model.device)
    input_mask = torch.tensor([input_mask], device=model.device)
    out_mask = torch.tensor([out_mask], device=model.device)
    with torch.no_grad():
        rep = model.forward2(input_ids, input_mask, out_mask)
    return rep.cpu().numpy()