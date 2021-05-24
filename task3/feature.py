import os, pickle
import numpy as np


class Feature(object):
    def __init__(self, sentid, tag, rep, type, frag_rep=None) -> None:
        if frag_rep is not None:
            self.frag_rep = frag_rep
        else:
            self.frag_rep = np.zeros(768)
        self.sentid = sentid    # sentence id
        self.tag = tag
        self.rep = rep      # 768 dim representation
        self.type = type    # Definition/Term/Refrential
        self.has_frag = False

    def __str__(self):
        s = f"sentence id: {self.sentid}\n"
        s += f"tag: {self.tag}\n"
        s += f"type: {self.type}\n"
        s += f"has_frag: {self.has_frag}\n"
        return s

class Pair(object):
    def __init__(self, feat1, feat2, relation) -> None:
        self.feat1 = feat1
        self.feat2 = feat2
        self.relation = relation
        self.same_sent = (feat1.sentid == feat2.sentid)

    def __str__(self) -> str:
        return str(self.feat1) + "\n" + str(self.feat2) + "\n" + f"relation: {self.relation}"