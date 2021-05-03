import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


data_path = "../../deft_corpus/data/deft_files/"
train_path = data_path + "train.pkl"
valid_path = data_path + "dev.pkl"
test_path = data_path + "test.pkl"

def load_data(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

train_x, train_y = load_data(train_path)
valid_x, valid_y = load_data(valid_path)
test_x, test_y = load_data(test_path)
str_train_x = [" ".join(lst) for lst in train_x]
str_valid_x = [" ".join(lst) for lst in valid_x]
str_test_x = [" ".join(lst) for lst in test_x]



tfidf = TfidfVectorizer(stop_words='english', decode_error='ignore')
train_x_counts_tf = tfidf.fit_transform(str_train_x)
valid_x_counts_tf = tfidf.transform(str_valid_x)
test_x_counts_tf = tfidf.transform(str_test_x)


models = {}
models["LR"] = LogisticRegression()
models["SVM"] = SVC()
models["CART"] = DecisionTreeClassifier()
models["MNB"] = MultinomialNB()
models["KNN"] = KNeighborsClassifier()
results = {}



for key in models:
    model = models[key]
    model.fit(train_x_counts_tf, train_y)
    predictions = model.predict(test_x_counts_tf)
    results[key] = classification_report(test_y, predictions)

for key in results:
    print(key)
    print(results[key])
    print()