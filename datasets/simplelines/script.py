import sys
from PIL import Image, ImageDraw
import random



def create_dataset(size):
    for i in range(size):
        with Image.open("datasets/simplelines/main.png") as im:
            draw = ImageDraw.Draw(im.resize(100, 100))
            draw.line(((0, random.randint(1, 100)), (random.randint(1, 100), 0)) + im.size, width=10, fill=128)    
            hr = im
            hr.save(f"datasets/simplelines/hr/{i}.png")


create_dataset(1000)
