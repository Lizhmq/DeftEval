import numpy as np
from numpy.lib.npyio import savez_compressed
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import os
from sklearn.metrics import classification_report


def save_pkl(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def process(pairs, type_dict=None, target_dict=None):
    xs, ys = [], []
    if type_dict is None:
        if not os.path.exists("type_dict.pkl"):
            type_dict = {}
            target_dict = {}
            for pair in pairs:
                if pair.relation not in target_dict:
                    target_dict[pair.relation] = len(target_dict)
                for feat in (pair.feat1, pair.feat2):
                    if feat.type not in type_dict:
                        type_dict[feat.type] = len(type_dict)
            save_pkl(type_dict, "type_dict.pkl")
            save_pkl(target_dict, "target_dict.pkl")
        else:
            type_dict = load_pkl("type_dict.pkl")
            target_dict = load_pkl("target_dict.pkl")
    for pair in pairs:
        x = []
        x += list(pair.feat1.rep + pair.feat1.frag_rep)
        type1 = [0] * len(type_dict)
        type1[type_dict[pair.feat1.type]] = 1
        x += type1

        x += list(pair.feat2.rep + pair.feat2.frag_rep)
        type2 = [0] * len(type_dict)
        type2[type_dict[pair.feat2.type]] = 1
        x += type2

        x += [int(pair.same_sent)]
        xs.append(x)
        ys.append(target_dict[pair.relation])
    return xs, ys



def main():
    train_path = "./save_feat.pkl"
    test_path = "./save_feat_test.pkl"
    predict_path = "./testdata.pkl"
    
    train_pos, train_neg = load_pkl(train_path)
    test_pos, test_neg = load_pkl(test_path)
    predict_data = load_pkl(predict_path)

    # train_x, train_y = process(train_pos + train_neg)
    type_dict = load_pkl("type_dict.pkl")
    target_dict = load_pkl("target_dict.pkl")
    # test_x, test_y = process(test_pos + test_neg, type_dict, target_dict)
    predict_x, predict_y = process(predict_data, type_dict, target_dict)

    # rf = RandomForestClassifier(n_estimators=100, n_jobs=20, verbose=3)
    # rf.fit(train_x, train_y)
    # joblib.dump(rf, "./random_forest.joblib2")
    rf = joblib.load("./random_forest.joblib2")
    # pred = rf.predict(test_x)
    # print(classification_report(test_y, pred))

    y = rf.predict(predict_x)
    save_pkl(y, "output.pkl")



if __name__ == "__main__":
    main()

