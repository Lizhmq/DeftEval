import math
import os
import pickle
import random
import torch
from tqdm import tqdm
from feature import Feature, Pair
from transformers import RobertaTokenizer
from model import gen_rep, build_model

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["x"], data["y"], data["bio"], \
            data["tag"], data["sistag"], data["target"]



class fake_args:
    def __init__(self) -> None:
        self.lambd = 0
        self.device = torch.device("cuda", 0)
        self.pretrain_dir = "../task2/save-1-crf/checkpoint-2000-0.4989"
        self.bio_size = 24

def feat_set(file_path, output_path="feat.pkl"):
    if os.path.exists(output_path):
        return pickle.load(open(output_path, "rb"))
    args = fake_args()
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_dir)
    model = build_model(args, args.bio_size, args.pretrain_dir)
    model.eval()

    data = read_pkl(file_path)
    type_dict = dict()      # d["T3"] = "Definition"
    rep_dict = dict()       # d["T3"] = np.1darray of dim 768, rep from bert
    sent_dict = dict()      # d["T3"] = 1
    pair_dict = dict()      # d[("T3", "T62")] = "Direct-Defines"
    for i, (x, y, bio_l, tag_l, sistag_l, target_l) in tqdm(enumerate(zip(*data))):
        tag_set = set(tag_l)
        valid_set = set(tag_set)
        for tag in tag_set:
            if tag.endswith("-1"):
                valid_set.discard(tag)
        tag_set = valid_set
        if len(tag_set) == 0:
            continue
        for tag in tag_set:
            mask = [tag == t for t in tag_l]
            rep = gen_rep(model, tokenizer, x.split(" "), mask)
            idx = mask.index(True)
            type = bio_l[idx][2:]      # remove "B-"
            sis_tag = sistag_l[idx]
            pair_target = target_l[idx]
            sent_dict[tag] = i
            type_dict[tag] = type
            rep_dict[tag] = rep
            if not sis_tag.endswith("0"):
                pair_dict[(tag, sis_tag)] = pair_target
    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump((type_dict, rep_dict, sent_dict, pair_dict), f)
    return type_dict, rep_dict, sent_dict, pair_dict


def testset(file_path, save_path=None):
    if os.path.exists(save_path):
        return pickle.load(open(save_path, "rb"))
    type_dict, rep_dict, sent_dict, pair_dict = feat_set(file_path)
    feat_dict = dict()
    tag_keys = type_dict.keys()
    useful_tags = []
    for tag in tag_keys:
        if tag.endswith("-frag"):
            continue
        useful_tags.append(tag)
        sentid = sent_dict[tag]
        rep = rep_dict[tag]
        type = type_dict[tag]
        feat_dict[tag] = Feature(sentid, tag, rep, type)
        fragtag = tag + "-frag"
        if fragtag in tag_keys:
            feat_dict[tag].frag_rep = rep_dict[fragtag]
            feat_dict[tag].has_frag = True
    out_pairs = []
    for tag1 in useful_tags:
        for tag2 in useful_tags:
            if tag2 == tag1:
                continue
            out_pairs.append(Pair(feat_dict[tag1], feat_dict[tag2], "None"))
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(out_pairs, f)
    return out_pairs


def create_dataset(file_path, save_path=None):
    type_dict, rep_dict, sent_dict, pair_dict = feat_set(file_path)
    feat_dict = dict()
    pos_pairs = []
    tag_keys = type_dict.keys()
    useful_tags = []
    for tag in tag_keys:
        if tag.endswith("-frag"):
            continue
        useful_tags.append(tag)
        sentid = sent_dict[tag]
        rep = rep_dict[tag]
        type = type_dict[tag]
        feat_dict[tag] = Feature(sentid, tag, rep, type)
        fragtag = tag + "-frag"
        if fragtag in tag_keys:
            feat_dict[tag].frag_rep = rep_dict[fragtag]
            feat_dict[tag].has_frag = True
    wrong = 0
    for (tag1, tag2), target in pair_dict.items():
        if tag1.endswith("-frag") or tag2.endswith("-frag"):
            continue
        if tag1 not in feat_dict or tag2 not in feat_dict:
            wrong += 1
            continue
        feat1 = feat_dict[tag1]
        feat2 = feat_dict[tag2]
        pos_pairs.append(Pair(feat1, feat2, target))
    pos_num = len(pos_pairs)
    neg_pairs = []
    for tag1 in useful_tags:
        curfile = tag1.split("deft")[0]
        tags = list(filter(lambda x: x.split("deft")[0] == curfile, useful_tags))
        tmptags = []
        for tag2 in tags:
            if tag2 == tag1:
                continue
            if (tag1, tag2) in pair_dict or (tag2, tag1) in pair_dict:
                continue
            tmptags.append(tag2)
        tmptags = random.sample(tmptags, min(3, len(tmptags)))
        for tag2 in tmptags:
            neg_pairs.append((tag1, tag2))
    
    # sample_ratio = 3 * int(math.sqrt(pos_num) + 1)
    # tag1_sample = random.sample(useful_tags, sample_ratio)
    # tag2_sample = random.sample(useful_tags, sample_ratio)
    # neg_pairs = []
    # for tag1 in tag1_sample:
    #     for tag2 in tag2_sample:
    #         if tag2 == tag1:
    #             continue
    #         if (tag1, tag2) in pair_dict or (tag2, tag1) in pair_dict:
    #             continue
    #         neg_pairs.append((tag1, tag2))
    # neg_pairs = random.sample(neg_pairs, pos_num)
    neg_pairs = [Pair(feat_dict[tag1], feat_dict[tag2], "None") for tag1, tag2 in neg_pairs]
    print(f"Error pairs: {wrong}")
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump((pos_pairs, neg_pairs), f)
    return pos_pairs, neg_pairs
