from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound
from .forms import SRImagesForm
from .models import SRImages


# Create your views here.


def say_hello(request):
    return HttpResponse('Hello pidor')


def upload_image(request):
    form = SRImagesForm(request.POST, request.FILES)
    if form.is_valid():
        form.save()
        return redirect('upload_image/')
    return HttpResponse('None')


def download_image(request):
    form = SRImagesForm()
    return render(request, '', {    # TODO: template 
        'form': form
    })
    

def page_not_found(request, exception):
    return HttpResponseNotFound('<h1> Page not found ;( <h1>')
    