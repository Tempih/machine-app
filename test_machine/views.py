import os
from sklearn import metrics
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
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
data = []
target = []
real_data = []
pos = []
neg = []
post = []
negt = []
rpos = []
rneg = []
path = os.getcwd()
directory = path+"/train/neg"
openf(directory, data, 0, neg)
directory = path+"/train/pos"
openf(directory, data, 1, pos)
# ras(pos,post,rpos)
#

datat = []
targett = []
real_datat = []
tpos = []
tneg = []
ttpos = []
ttneg = []
trpos = []
trneg = []
directory = path+"/test/neg"
openf(directory, data, 0, neg)
directory = path+"/test/pos"
openf(directory, data, 1, pos)
ras(data,target,real_data)
ras(pos,tpos,ttpos)
ras(neg,tneg,ttneg)
# ras(datat,targett,real_datat)
# ras(tpos,ttpos,trpos)
# ras(tneg,ttneg,trneg)

# data = []
# f = open("/Users/artemmihajlov/PycharmProjects/untitled10/labeledBow.feat","r")
# for line in f:
#     data.append(line)
# f.close()
# target = []
# random.shuffle(data)
# for i in range(len(data)):
#     if data[i][1] == ' ':
#         target.append(data[i][0])
#         data[i] = data[i].replace(data[i][0], '', 1)
#
#     else:
#         target.append(data[i][0]+data[i][1])
#         data[i] = data[i].replace(data[i][0]+data[i][1], '', 1)
#
# test = []
# for i in data:
#     z = [0]*89527
#     for j in i.split():
#         m = j.split(":")
#         z[int(m[0])] = int(m[1])
#     test.append(z)
# vect = CountVectorizer(max_features = 3000)
# vect.fit(real_data)
# train_data = vect.transform(real_data)
# test_data = vect.transform(real_datat)
# tfidf = TfidfTransformer(use_idf = True).fit(train_data)
# train_data_tfidf = tfidf.transform(train_data)
# test_data_tfidf = tfidf.transform(test_data)
#
# clf = DecisionTreeClassifier().fit(train_data_tfidf, target)
# prediction = clf.predict(test_data_tfidf)
# print(metrics.classification_report(targett, prediction))
# print(metrics.confusion_matrix(targett, prediction), '\n')
# print((metrics.accuracy_score(targett, prediction)) * 100)
X_train, X_test, y_train, y_test = train_test_split(real_data, target, test_size=0.2, random_state=42)
vect = CountVectorizer(max_features = 50000)
vect.fit(X_train)
train_data = vect.transform(X_train)
test_data = vect.transform(X_test)
tfidf = TfidfTransformer(use_idf = True).fit(train_data)
train_data_tfidf = tfidf.transform(train_data)
test_data_tfidf = tfidf.transform(test_data)
X_train, X_test, y_train_p, y_test_p = train_test_split(ttneg, tneg, test_size=0.2, random_state=42)
vect_p = CountVectorizer(max_features = 50000)
vect_p.fit(X_train)
train_data = vect_p.transform(X_train)
test_data = vect_p.transform(X_test)
tfidf_p = TfidfTransformer(use_idf = True).fit(train_data)
train_data_tfidf_p = tfidf_p.transform(train_data)
test_data_tfidf_p = tfidf_p.transform(test_data)
X_train, X_test, y_train_n, y_test_n = train_test_split(ttpos, tpos, test_size=0.2, random_state=42)
vect_n = CountVectorizer(max_features = 50000)
vect_n.fit(X_train)
train_data = vect_n.transform(X_train)
test_data = vect_n.transform(X_test)
tfidf_n = TfidfTransformer(use_idf = True).fit(train_data)
train_data_tfidf_n = tfidf_n.transform(train_data)
test_data_tfidf_n = tfidf_n.transform(test_data)
print(2)
clf_main = LogisticRegression(max_iter=500, C=10)
clf_main = clf_main.fit(train_data_tfidf, y_train)
prediction = clf_main.predict(test_data_tfidf)
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction), '\n')
print((metrics.accuracy_score(y_test, prediction)) * 100)

print(2)
clf_p = LogisticRegression(max_iter=500, C=10)
clf_p = clf_p.fit(train_data_tfidf_n, y_train_n)
prediction = clf_p.predict(test_data_tfidf_n)
print(metrics.classification_report(y_test_n, prediction))
print(metrics.confusion_matrix(y_test_n, prediction), '\n')
print((metrics.accuracy_score(y_test_n, prediction)) * 100)

print(2)
clf_n = LogisticRegression(max_iter=500, C=10)
clf_n = clf_n.fit(train_data_tfidf_p, y_train_p)
prediction = clf_n.predict(test_data_tfidf_p)
print(metrics.classification_report(y_test_p, prediction))
print(metrics.confusion_matrix(y_test_p, prediction), '\n')
print((metrics.accuracy_score(y_test_p, prediction)) * 100)
print(2)
from sklearn.model_selection import cross_validate

# classificator = LogisticRegression(random_state=0)
# classificator.fit(train_data_tfidf, target)
#
# print(classificator)
# #

from django.shortcuts import render
from .forms import UserForm

def index(request):
  submitbutton= request.POST.get("submit")

  status=''
  star = ''
  form= UserForm(request.POST or None)
  if form.is_valid():
        review= ['''{0}'''.format(form.cleaned_data.get("review"))]
        test_data = vect.transform(review)
        test_data_tfidf = tfidf.transform(test_data)
        mresult = clf_main.predict(test_data_tfidf)
        if mresult[0] == 0:
            status = "Negative"
            test_data_n = vect_p.transform(review)
            test_data_tfidf_n = tfidf_p.transform(test_data_n)
            vresult = clf_n.predict(test_data_tfidf_n)
            star = vresult[0]
        else:
            status = "Positive"
            test_data_p = vect_n.transform(review)
            test_data_tfidf_p = tfidf_n.transform(test_data_p)
            vresult = clf_p.predict(test_data_tfidf_p)
            star = vresult[0]
  context= {'form': form, 'review': status, "star":star,
            'submitbutton': submitbutton,
            }

  return render(request, 'index.html', context)