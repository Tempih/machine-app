import os
from sklearn import metrics
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
def openf(directory,data,num , table):
    files = os.listdir(directory)
    for file in files:
        f = open(directory + '/' + file, "r")
        test = file.split("_")
        text = f.read()
        table.append([text, int(test[1].split(".")[0])])
        data.append([text, num])
        f.close()
def ras(data,target,real_data):
    random.shuffle(data)
    for i in range(len(data)):
        target.append(data[i][1])
        del data[i][1]
        real_data.append(data[i][0])

random.seed(42)
data = []
target = []
real_data = []
pos = []
neg = []
tpos = []
tneg = []
ttpos = []
ttneg = []
directory = "/Users/artemmihajlov/PycharmProjects/untitled10/django_example/train/neg"
openf(directory, data, 0, neg)
directory = "/Users/artemmihajlov/PycharmProjects/untitled10/django_example/train/pos"
openf(directory, data, 1, pos)
directory = "/Users/artemmihajlov/PycharmProjects/untitled10/django_example/test/neg"
openf(directory, data, 0, neg)
directory = "/Users/artemmihajlov/PycharmProjects/untitled10/django_example/test/pos"
openf(directory, data, 1, pos)
ras(data, target, real_data)
ras(pos, tpos, ttpos)
ras(neg, tneg, ttneg)

X_train, X_test, y_train, y_test = train_test_split(real_data, target, test_size=0.2, random_state=42)
vect = CountVectorizer(max_features=50000)
vect.fit(X_train)
train_data = vect.transform(X_train)
test_data = vect.transform(X_test)
tfidf = TfidfTransformer(use_idf=True).fit(train_data)
train_data_tfidf = tfidf.transform(train_data)
test_data_tfidf = tfidf.transform(test_data)
clf = SVC()
clf = clf.fit(train_data_tfidf, y_train)
prediction = clf.predict(test_data_tfidf)
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction), '\n')
print((metrics.accuracy_score(y_test, prediction)) * 100)

X_train, X_test, y_train, y_test = train_test_split(ttpos, tpos, test_size=0.2, random_state=42)
vect = CountVectorizer(max_features=50000)
vect.fit(X_train)
train_data = vect.transform(X_train)
test_data = vect.transform(X_test)
tfidf = TfidfTransformer(use_idf=True).fit(train_data)
train_data_tfidf = tfidf.transform(train_data)
test_data_tfidf = tfidf.transform(test_data)
clf = SVC()
clf = clf.fit(train_data_tfidf, y_train)
prediction = clf.predict(test_data_tfidf)
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction), '\n')
print((metrics.accuracy_score(y_test, prediction)) * 100)

X_train, X_test, y_train, y_test = train_test_split(ttneg, tneg, test_size=0.2, random_state=42)
vect = CountVectorizer(max_features=50000)
vect.fit(X_train)
train_data = vect.transform(X_train)
test_data = vect.transform(X_test)
tfidf = TfidfTransformer(use_idf=True).fit(train_data)
train_data_tfidf = tfidf.transform(train_data)
test_data_tfidf = tfidf.transform(test_data)
clf = SVC()
clf = clf.fit(train_data_tfidf, y_train)
prediction = clf.predict(test_data_tfidf)
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction), '\n')
print((metrics.accuracy_score(y_test, prediction)) * 100)

