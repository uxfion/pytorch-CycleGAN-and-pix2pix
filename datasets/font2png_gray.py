import os
import random
from PIL import ImageFont, Image, ImageDraw

font_name = "pacifico"
ttf_path = f"./{font_name}.ttf"
text_size = 400
font = ImageFont.truetype(ttf_path, text_size)

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

if not os.path.exists(f"gray_font_{font_name}"):
    os.makedirs(f"gray_font_{font_name}")

for char in characters:
    img = Image.new("L", (1200, 900), 255)
    draw = ImageDraw.Draw(img)
    white = 0
    text_width = font.getbbox(char)[2]
    text_height = font.getbbox(char)[3]
    # mov center
    x = (1200 - text_width) // 2
    y = (900 - text_height - 200) // 2
    draw.text((x, y), char, font=font, fill=white)
    name = char
    name = "".join([c + "_" if c.islower() else c for c in name])
    img.save(f"gray_font_{font_name}/{name}.png")

# 随意组合两个大小写字母
for char1 in characters[:52]:
    for char2 in characters[:52]:
        img = Image.new("L", (1200, 900), 255)
        draw = ImageDraw.Draw(img)
        white = 0
        string = char1 + char2
        text_width = font.getbbox(string)[2]
        text_height = font.getbbox(string)[3]
        # mov center
        x = (1200 - text_width) // 2
        y = (900 - text_height - 200) // 2
        draw.text((x, y), string, font=font, fill=white)
        name = char1 + char2
        name = "".join([c + "_" if c.islower() else c for c in name])
        img.save(f"gray_font_{font_name}/{name}.png")

# 随机随意组合四个大小写字母
for i in range(2000):
    img = Image.new("L", (1500, 900), 255)
    draw = ImageDraw.Draw(img)
    white = 0
    string = ""
    for j in range(4):
        char = random.choice(characters[:52])
        string += char
    text_width = font.getbbox(string)[2]
    text_height = font.getbbox(string)[3]
    # mov center
    x = (1200 - text_width) // 2
    y = (900 - text_height - 200) // 2
    draw.text((x, y), string, font=font, fill=white)
    name = string[0] + string[1] + string[2] + string[3]
    name = "".join([c + "_" if c.islower() else c for c in name])
    img.save(f"gray_font_{font_name}/{name}.png")
