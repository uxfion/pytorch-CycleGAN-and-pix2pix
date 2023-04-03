import os
from PIL import ImageFont, Image, ImageDraw

font_name = "times"
ttf_path = f"./{font_name}.ttf"
text_size = 400
font = ImageFont.truetype(ttf_path, text_size)

strings = [
    "z",
    "u",
    "s",
    "t",
    "zu",
    "zU",
    "Zu",
    "ZU",
    "zust",
    "Zust",
    "ZUST",
    "Chm",
    "Lyh",
]

if not os.path.exists(f"test_gray_font_{font_name}"):
    os.makedirs(f"test_gray_font_{font_name}")

for string in strings:
    img = Image.new("L", (1500, 900), 255)
    draw = ImageDraw.Draw(img)
    white = 0
    text_width = font.getbbox(string)[2]
    text_height = font.getbbox(string)[3]
    x = (1500 - text_width) // 2
    y = (900 - text_height - 200) // 2
    draw.text((x, y), string, font=font, fill=white)
    name = string
    name = "".join([c + "_" if c.islower() else c for c in name])
    img.save(f"test_gray_font_{font_name}/{name}.png")
