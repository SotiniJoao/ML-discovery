import pandas as pd
import numpy as np
import os
import random

folder = 'C:/Users/jl_sa/Desktop/UTK/images and landmarks/UTKFace'

names = [f for f in os.listdir(folder)]

images = []

for i in range(0,9000):
    images.append(names[random.randrange(0,23707)])
