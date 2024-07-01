from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse
from .forms import SRImagesForm
from .models import SRImages
from django.template.loader import render_to_string
from django.shortcuts import render
import os
from django.conf import settings

# Create your views here.

menu = [
    {'title' : 'App', 'url_name': 'app'},
    {'title' : 'About', 'url_name': 'about'},
    {'title' : 'Help', 'url_name': 'help'},
]

def index(request):
    data = {
        'title': 'Main Page',
        'menu': menu,
    }
    return render(request, 'index.html', context=data)



def help(request):
    return render(request, 'help.html')


def about(request):
    return render(request, 'about.html')


def app(request):
    if request.method == 'POST':
        form = SRImagesForm(request.POST, request.FILES)
        print(request.FILES)
        if form.is_valid():
            old_images = SRImages.objects.all()
            for image in old_images:
                if os.path.exists(os.path.join(settings.MEDIA_ROOT, image.image.name)):
                    os.remove(os.path.join(settings.MEDIA_ROOT, image.image.name))
                image.delete()
            # TODO: model output
            uploaded_image = form.save()
            return JsonResponse({'image_url': uploaded_image.image.url})
        
    else:
        form = SRImagesForm()

    images = SRImages.objects.all()
    last_image = images.last() if images.exists() else None
    return render(request, 'app.html', {'form': form, 'image': last_image})


    

def page_not_found(request, exception):
    return HttpResponseNotFound('<h1> Page not found ;( <h1>')
    