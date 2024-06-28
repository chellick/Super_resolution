from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound
from .forms import SRImagesForm
from .models import SRImages
from django.template.loader import render_to_string
from django.shortcuts import render

# Create your views here.



def index(request):
    data = {
        'title': 'Main page'
    }
    return render(request, 'index.html', context=data)



def help(request):
    return render(request, 'help.html')


def about(request):
    return render(request, 'about.html')


def app(request):
    return render(request, 'app.html')



# def upload_image(request):
#     form = SRImagesForm(request.POST, request.FILES)
#     if form.is_valid():
#         form.save()
#         return redirect('upload_image/')
#     return HttpResponse('None')


# def download_image(request):
#     form = SRImagesForm()
#     return render(request, '', {    # TODO: template 
#         'form': form
#     })
    

def page_not_found(request, exception):
    return HttpResponseNotFound('<h1> Page not found ;( <h1>')
    