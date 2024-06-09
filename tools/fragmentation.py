import numpy as np
import matplotlib.pyplot as plt



def Fragmentation(path, pixels):
    image = plt.imread(path)
    chunks = []
    
    width, height, depth = image.shape
    
    for w in range(width//pixels):
        for h in range(height//pixels):
            chunks.append(image[pixels*w:pixels*(w+1), pixels*h:pixels*(h+1)])
    
    return chunks

    
