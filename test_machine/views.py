import joblib
from django.shortcuts import render
from .forms import UserForm
path = ('/home/Tempiha/machine-app/')
vect = joblib.load(path+'vect_main.sav')
tfidf = joblib.load(path+'tfidf_main.sav')
vect_p = joblib.load(path+'vect_pos.sav')
tfidf_p = joblib.load(path+'tfidf_pos.sav')
vect_n = joblib.load(path+'vect_neg.sav')
tfidf_n = joblib.load(path+'tfidf_neg.sav')
clf_main = joblib.load(path+'main.sav')
clf_p = joblib.load(path+'pos.sav')
clf_n = joblib.load(path+'neg.sav')

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