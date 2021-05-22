import os
import pickle


def out(strs, path):
    with open(path, "w") as f:
        f.writelines(strs)


def main():
    path1 = "../../deft_corpus/data/deft_files/test2.pkl"
    path2 = "./test-aug-3500.txt"
    output_1 = "../input/ref/task_2.deft"
    output_2 = "../input/res/task_2.deft"

    datas = pickle.load(open(path1, "rb"))
    xs, ys, bios = datas["x"], datas["y"], datas["bio"]
    
    with open(path2, "r") as f:
        ls = f.readlines()
    ls = [list(st.split(" ")) for st in ls]
    assert len(ls) == len(bios)


    label_list = ["O", "B-Term", "I-Term", "B-Definition", \
                "I-Definition", "B-Alias-Term", "I-Alias-Term", \
                "B-Referential-Definition", "I-Referential-Definition", \
                "B-Referential-Term", "I-Referential-Term", "B-Qualifier", "I-Qualifier"]
    def m(tok):
        if tok in label_list:
            return tok
        return "O"

    ref_lines, res_lines = [], []
    for i, (x, ref, res) in enumerate(zip(xs, bios, ls)):
        x = x.replace("\n", "")
        x = x.split(" ")
        assert len(ref) >= len(res)
        if len(ref) > len(res):
            print(len(ref) - len(res))
        for j in range(len(res)):
            res[j] = m(res[j])
            ref_lines.append("\t".join([x[j], "path", "0", "0", ref[j]]) + "\n")
            res_lines.append("\t".join([x[j], "path", "0", "0", res[j]]) + "\n")
        for j in range(len(res), len(ref)):
            ref_lines.append("\t".join([x[j], "path", "0", "0", ref[j]]) + "\n")
            res_lines.append("\t".join([x[j], "path", "0", "0", "O"]) + "\n")
        ref_lines.append("\n\n")
        res_lines.append("\n")
    out(ref_lines, output_1)
    out(res_lines, output_2)



if __name__ == "__main__":
    main()