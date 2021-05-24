import os, pickle
import re

def read_file(filename):
    xs, ys, bios = [], [], []               # for task1, 2
    tags, sistags, targets = [], [], []     # for task3
    with open(filename) as f:
        lines = list(f.readlines())
        sentence = ""
        bio_lst, tag_lst, sistag_lst, target_lst = [], [], [], []
        hasdef = 0
        for idx, line in enumerate(lines):
            line = lines[idx].strip().split("\t")
            if len(line) == 1:
                if len(sentence) > 0 and (not re.match(r'^\s*\d+\s*\.$', sentence)
                                              or lines[idx - 1] == '\n'):
                    xs.append(sentence)
                    ys.append(hasdef)
                    bios.append(bio_lst)
                    tags.append(tag_lst)
                    sistags.append(sistag_lst)
                    targets.append(target_lst)
                    sentence = ""
                    hasdef = 0
                    bio_lst, tag_lst, sistag_lst, target_lst = [], [], [], []
                continue
            # if len(line) != 8:
            #     print(line, len(line))
            # assert(len(line) == 8)
            if len(line) == 8:
                tok, file, bio = line[0], line[1][1:], line[4][1:]     # remove additional space
                tag, sistag, target = line[5][1:], line[6][1:], line[7][1:]
                if sentence != "":
                    sentence += " "
                sentence += tok
                bio_lst.append(bio)
                tag_lst.append(file.split("/")[-1] + tag)
                sistag_lst.append(file.split("/")[-1] + sistag)
                target_lst.append(target)
                if bio[2:] == "Definition":
                    hasdef = 1
            else:
                assert(len(line) == 5)
                tok, bio = line[0], line[4][1:]
                if sentence != "":
                    sentence += " "
                sentence += tok
                bio_lst.append(bio)
                if bio[2:] == "Definition":
                    hasdef = 1
        if len(sentence) > 0:
            xs.append(sentence)
            ys.append(hasdef)
            bios.append(bio_lst)
            tags.append(tag_lst)
            sistags.append(sistag_lst)
            targets.append(target_lst)
    return (xs, ys, bios, tags, sistags, targets)

def read_dir(path):
    allx, ally, allbio, alltag, allsis, alltarget = [], [], [], [], [], []
    for file in sorted(os.listdir(path)):
        if file.endswith(".deft"):
            p = os.path.join(path, file)
            xs, ys, bios, tags, sistags, targets = read_file(p)
            allx.extend(xs)
            ally.extend(ys)
            allbio.extend(bios)
            alltag.extend(tags)
            allsis.extend(sistags)
            alltarget.extend(targets)
    dic = {
        "x": allx,
        "y": ally,
        "bio": allbio,
        "tag": alltag,
        "sistag": allsis,
        "target": alltarget
    }
    return dic

def read_1(path):
    xs, ys = [], []
    for file in sorted(os.listdir(path)):
        if file.endswith(".deft"):
            p = os.path.join(path, file)
            with open(p, "r") as f:
                for line in f.readlines():
                    x, y = line.strip().split("\t")
                    x = x[1:-1]
                    y = int(y[1:-1])
                    xs.append(x)
                    ys.append(y)
    dic = {"x": xs, "y": ys}
    return dic


def save_pkl(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj



if __name__ == "__main__":
    train_path = "../deft_corpus/data/deft_files/train"
    dev_path = "../deft_corpus/data/deft_files/dev"
    test_path_1 = "../deft_corpus/data/test_files/labeled/subtask_1"
    test_path_2 = "../deft_corpus/data/test_files/labeled/subtask_2"
    test_path_3 = "../deft_corpus/data/test_files/labeled/subtask_3"
    
    save_path_t = "../deft_corpus/data/deft_files/nall_train.pkl"
    save_path_d = "../deft_corpus/data/deft_files/nall_dev.pkl"
    # save_path_t1 = "../deft_corpus/data/deft_files/test1.pkl"
    save_path_t2 = "../deft_corpus/data/deft_files/ntest2.pkl"
    save_path_t3 = "../deft_corpus/data/deft_files/ntest3.pkl"

    dic = read_dir(train_path)
    save_pkl(dic, save_path_t)

    dic = read_dir(dev_path)
    save_pkl(dic, save_path_d)

    # dic = read_1(test_path_1)
    # save_pkl(dic, save_path_t1)
    dic = read_dir(test_path_2)
    save_pkl(dic, save_path_t2)
    dic = read_dir(test_path_3)
    save_pkl(dic, save_path_t3)
    