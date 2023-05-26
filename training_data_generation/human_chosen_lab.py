import tkinter as tk
from random import randint
from skimage import color
import numpy as np

colors = {'white': (255, 255, 255), 'gray': (127, 127, 127), 'black': (0, 0, 0), 'red': (255, 0, 0),
          'pink': (255, 122, 180), 'dark red': (59, 0, 14), 'orange': (255, 111, 0), 'brown': (61, 27, 0),
          'yellow': (255, 242, 0), 'green': (0, 255, 0), 'dark green': (3, 46, 11), 'teal': (12, 153, 111),
          "light blue": (135, 255, 255), 'blue': (0, 0, 255), 'dark blue': (3, 5, 102), 'purple': (80, 13, 143)}

def rgb_to_lab(rgb):
    rgb = [x / 255 for x in rgb]
    return color.rgb2lab(np.array([[rgb]])).flatten()

def save_color(color_index):
    global random_color
    with open('colors.txt', 'a') as f:
        lab = rgb_to_lab(random_color)
        f.write(f'{lab[0]} {lab[1]} {lab[2]} {color_index}\n')

def on_color_click(color_index):
    save_color(color_index)
    update_random_color()

def update_random_color():
    global random_color
    random_color = [randint(0, 255) for _ in range(3)]
    canvas.config(bg=f'#{random_color[0]:02x}{random_color[1]:02x}{random_color[2]:02x}')

root = tk.Tk()
root.title('Color Picker')

random_color = None
color_label = tk.Label(root, text='Color to match:', font=('Arial', 24))
canvas = tk.Canvas(root,)
canvas.grid(row=0, column=0, columnspan=4)

update_random_color()

for i, color_name in enumerate(colors):
    btn = tk.Button(root,
                    text=color_name,
                    bg=f'#{colors[color_name][0]:02x}{colors[color_name][1]:02x}{colors[color_name][2]:02x}',
                    command=lambda i=i: on_color_click(i),
                    width=15,
                    height=10)
    btn.grid(row=i//4+1,column=i%4)

root.mainloop()