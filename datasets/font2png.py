import os
from PIL import ImageFont, Image, ImageDraw

font_name = "pacifico"
ttf_path = f"./{font_name}.ttf"
text_size = 400
font = ImageFont.truetype(ttf_path, text_size)

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

if not os.path.exists(f"font_{font_name}"):
    os.makedirs(f"font_{font_name}")

for char in characters:
    img = Image.new("RGB", (1200, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    white = (0, 0, 0)
    text_width = font.getbbox(char)[2]
    text_height = font.getbbox(char)[3]
    x = (1200 - text_width) // 2
    y = (900 - text_height - 200) // 2
    draw.text((x, y), char, font=font, fill=white)
    name = char
    if char.islower():
        name += "_lower"
    img.save(f"font_{font_name}/{name}.png")

# 随意组合两个大小写字母
for char1 in characters[:52]:
    for char2 in characters[:52]:
        img = Image.new("RGB", (1200, 900), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        white = (0, 0, 0)
        string = char1 + char2
        text_width = font.getbbox(string)[2]
        text_height = font.getbbox(string)[3]
        x = (1200 - text_width) // 2
        y = (900 - text_height - 200) // 2
        draw.text((x, y), string, font=font, fill=white)
        name1 = char1
        name2 = char2
        if char1.islower():
            name1 += "_lower"
        if char2.islower():
            name2 += "_lower"
        img.save(f"font_{font_name}/{name1}{name2}.png")
