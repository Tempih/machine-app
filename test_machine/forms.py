# from django import forms
#
#
# class UserForm(forms.Form):
#     name = forms.CharField()
#     age = forms.IntegerField()
from django import forms
class UserForm(forms.Form):
     review = forms.CharField(widget=forms.Textarea, label='')
     required_css_class = "field"
