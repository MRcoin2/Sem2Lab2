import random
from math import sqrt

import numpy as np
from skimage import color

categories = ['white', 'gray', 'black', 'red', 'pink', 'dark red', 'orange', 'brown', 'yellow', 'green', 'dark green',
          'teal', "light blue", 'blue', 'dark blue', 'purple']

colors = [((float(color.split()[0]), float(color.split()[1]), float(color.split()[2])), int(color.split()[3])) for color
          in open(r"colors.txt").readlines()]

def distance(color1, color2):
    return sqrt((color2[0] - color1[0]) ** 2 + (color2[1] - color1[1]) ** 2 + (color2[2] - color1[2]) ** 2)


def random_rgb():
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

def rgb_to_lab(rgb):
    rgb = [x / 255 for x in rgb]
    return color.rgb2lab(np.array([[rgb]])).flatten()

def save_category(color, category):
    with open("trainingv4.txt", "a") as f:
        f.write(f"{color[0]/100} {color[1]/128} {color[2]/128} {category}\n")


def assign_category():
    random_color = rgb_to_lab(random_rgb())
    distances = []
    for color in colors:
        distances.append((distance(color[0], (random_color[0],random_color[1],random_color[2])), color[1]))
    distances.sort()
    category = distances[0][1]
    save_category(random_color, category)


while True:
    assign_category()
