import os
from django.shortcuts import render
from .forms import UserForm
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def openf(directory, data, num, table):
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
tpos = []
tneg = []
ttpos = []
ttneg = []
path = os.getcwd()
directory = path + "/train/neg"
openf(directory, data, 0, neg)
directory = path + "/train/pos"
openf(directory, data, 1, pos)
directory = path + "/test/neg"
openf(directory, data, 0, neg)
directory = path + "/test/pos"
openf(directory, data, 1, pos)
ras(data, target, real_data)
ras(pos, tpos, ttpos)
ras(neg, tneg, ttneg)

X_train, X_test, y_train, y_test = train_test_split(real_data, target, test_size=0.2, random_state=42)
vect = CountVectorizer(max_features=50000)
vect.fit(X_train)
train_data = vect.transform(X_train)
tfidf = TfidfTransformer(use_idf=True).fit(train_data)
train_data_tfidf = tfidf.transform(train_data)

X_train, X_test, y_train_p, y_test_p = train_test_split(ttpos, tpos, test_size=0.2, random_state=42)
vect_p = CountVectorizer(max_features=50000)
vect_p.fit(X_train)
train_data = vect_p.transform(X_train)
tfidf_p = TfidfTransformer(use_idf=True).fit(train_data)
train_data_tfidf_p = tfidf_p.transform(train_data)

X_train, X_test, y_train_n, y_test_n = train_test_split(ttneg, tneg, test_size=0.2, random_state=42)
vect_n = CountVectorizer(max_features=50000)
vect_n.fit(X_train)
train_data = vect_n.transform(X_train)
tfidf_n = TfidfTransformer(use_idf=True).fit(train_data)
train_data_tfidf_n = tfidf_n.transform(train_data)

clf_main = LogisticRegression(max_iter=500, C=10, solver='liblinear')
clf_main = clf_main.fit(train_data_tfidf, y_train)

clf_p = LogisticRegression(max_iter=800, C=10, solver='liblinear')
clf_p = clf_p.fit(train_data_tfidf_n, y_train_n)

clf_n = LogisticRegression(max_iter=800, C=10, solver='liblinear')
clf_n = clf_n.fit(train_data_tfidf_p, y_train_p)


def index(request):
  submitbutton = request.POST.get("submit")
  status = ''
  star = ''
  form = UserForm(request.POST or None)
  if form.is_valid():
        review = ['''{0}'''.format(form.cleaned_data.get("review"))]
        test_data = vect.transform(review)
        test_data_tfidf = tfidf.transform(test_data)
        mresult = clf_main.predict(test_data_tfidf)
        if mresult[0] == 0:
            status = "Negative"
            test_data_n = vect_n.transform(review)
            test_data_tfidf_n = tfidf_n.transform(test_data_n)
            vresult = clf_n.predict(test_data_tfidf_n)
            star = vresult[0]
        else:
            status = "Positive"
            test_data_p =vect_p.transform(review)
            test_data_tfidf_p = tfidf_p .transform(test_data_p)
            vresult = clf_p.predict(test_data_tfidf_p)
            star = vresult[0]
  context = {'form': form, 'review': status, "star":star,
            'submitbutton': submitbutton,
            }

  return render(request, 'index.html', context)