import sys
from PIL import Image, ImageDraw
import random


def create_dataset(size):
    for i in range(size):
        with Image.open("datasets/simplelines/main.png") as im:
            draw = ImageDraw.Draw(im)
            draw.line(((0, random.randint(1, im.size[0])), (random.randint(1, im.size[1]), 0)) + im.size, width=10, fill=128)    
            hr = im
            lr = im.resize([im.size[0]// 10, im.size[1]// 10])
            hr.save(f"datasets/simplelines/hr/{i}.png")
            lr.save(f"datasets/simplelines/lr/{i}.png")

create_dataset(1000)
