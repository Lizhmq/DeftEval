## SemEval Task6 - DeftEval

[SemEval-2020](https://www.aclweb.org/portal/content/semeval-2020-international-workshop-semantic-evaluation) is an International Workshop on Semantic Evaluation hold by ACL.

In this repository, we focus on the SemEval Task6 -- [DeftEval](https://competitions.codalab.org/competitions/20900): Extracting term-definition pairs in free text.

### Overview

DeftEval is split into three subtasks:

**Subtask 1: Sentence Classification**

Given a sentence, classify whether or not it contains a definition. This is the traditional definition extraction task.

**Subtask 2: Sequence Labeling**

Label each token with BIO tags according to the corpus' tag specification.

**Subtask 3: Relation Classification**

Given the tag sequence labels, label the relations between each tag according to the corpus' relation specification.

Please refer to the offered link for more information.

### Subtask-1: Sentence Classification

As mentioned above, this subtask is a sentence binary classification problem.

We implement the following traditional ML methods and DL methods for demonstration:

* Traditional ML: Naive Bayesian, KNN, Decision Tree, Logistic Regression, SVM
* Deep Learning: LSTM, BERT + fine-tuning

The results are listed in [res.txt](task1/res.txt):

|          | **NB** | **KNN** | **CART** | **LR** | **SVM** | **LSTM** | **BERT** |
| -------- | ------ | ------- | -------- | ------ | ------- | -------- | -------- |
| Accuracy | 71%    | 68%     | 71%      | 73%    | 77%     | 81%      | 86%      |
| F1       | 0.30   | 0.03    | 0.52     | 0.44   | 0.53    | 0.69     | 0.79     |
| Macro F1 | 0.56   | 0.42    | 0.66     | 0.63   | 0.69    | 0.78     | 0.84     |

(Note that the hyper-parameters are not searched/selected well. I just do a simple experiment. :)

### Final Report

Please refer to the pdf file "DeftEval - 李拙.pdf". (Updated on May 26, 2021)
