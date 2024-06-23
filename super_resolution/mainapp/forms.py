from django import forms
from .models import SRImages

# Create forms here

class SRImagesForm(forms.ModelForm):
    class Meta:
        model = SRImages
        fields = [
            'image'
        ]
    
    
