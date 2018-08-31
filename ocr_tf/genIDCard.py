# -*- coding: utf-8 -*-
"""
身份证文字+数字生成类
"""

import numpy as np
import copy
import random
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
# from freetype

text = u"W test 123"
im = Image.new("RGB", (300, 50), (255, 255, 255))
dr = ImageDraw.Draw(im)
font = ImageFont.truetype(os.path.join("fonts", "OCR-B.ttf"), 40, encoding="utf-8")

dr.text((10, 5), text, font=font, fill="#000000")

# im.show()
# im.save("t.png")

def gen_id_card():
    pass

def gen_text(self, is_ran=False):
    text = ''
    vecs = np.zeros((self.max_size * self.len))

    if is_ran == True:
        size = random.randint(1, self.max_size)
    else:
        size = self.max_size

    for i in range(size):
        c = random.choice(self.char.set)
        vec = self.char2ve