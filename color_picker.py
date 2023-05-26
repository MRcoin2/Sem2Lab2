import numpy as np
import pygame
import sys
import pygame_gui

from pygame_gui.elements import UIButton
from pygame_gui.windows import UIColourPickerDialog
from skimage import color

pygame.init()

pygame.display.set_caption('Colour Picking App')
SCREEN = pygame.display.set_mode((800, 600))

ui_manager = pygame_gui.UIManager((800, 600))
background = pygame.Surface((800, 600))
background.fill("#3a3b3c")
colour_picker_button = UIButton(relative_rect=pygame.Rect(-180, -60, 150, 30),
                                text='Pick Colour',
                                manager=ui_manager,
                                anchors={'left': 'right',
                                         'right': 'right',
                                         'top': 'bottom',
                                         'bottom': 'bottom'})
colour_picker = None
current_colour = pygame.Color(0, 0, 0)
picked_colour_surface = pygame.Surface((400, 400))
# text_box = pygame_gui.elements.UITextBox("", pygame.Rect(200, 500, 400, 100), ui_manager, )
text_box = pygame_gui.elements.UITextEntryLine(pygame.Rect(200, 500, 400, 100), ui_manager, )
picked_colour_surface.fill(current_colour)

clock = pygame.time.Clock()


def rgb_to_lab(rgb):
    rgb = [x / 255 for x in rgb]
    return color.rgb2lab(np.array([[rgb]])).flatten()




while True:
    time_delta = clock.tick(60) / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == colour_picker_button:
            colour_picker = UIColourPickerDialog(pygame.Rect(160, 50, 420, 400),
                                                 ui_manager,
                                                 window_title="Change Colour...",
                                                 initial_colour=current_colour)
            colour_picker_button.disable()
        if event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED:
            current_colour = event.colour
            lab_color = rgb_to_lab([current_colour.r, current_colour.g, current_colour.b])
            # text_box.set_text(f"{lab_color[0]} {lab_color[1]} {lab_color[2]}")
            # text_box.rebuild()
            text_box.text = f"{lab_color[0]/100} {lab_color[1]/128} {lab_color[2]/128}"
            text_box.rebuild()
            picked_colour_surface.fill(current_colour)
        if event.type == pygame_gui.UI_WINDOW_CLOSE:
            colour_picker_button.enable()
            colour_picker = None

        ui_manager.process_events(event)

    ui_manager.update(time_delta)

    SCREEN.blit(background, (0, 0))
    SCREEN.blit(picked_colour_surface, (200, 100))

    ui_manager.draw_ui(SCREEN)

    pygame.display.update()
