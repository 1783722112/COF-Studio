# -*- coding: utf-8 -*-
"""生成软件图标 cof.ico —— 示意图：一个六边形孔道 + 六个节点（COF 的经典意象）。"""
import math
from PIL import Image, ImageDraw

S = 1024          # 先画大图再缩，边缘才平滑
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆角底板（从深蓝到近黑的竖向渐变）
pad = 40
r = 190
bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bd = ImageDraw.Draw(bg)
for y in range(pad, S - pad):
    t = (y - pad) / float(S - 2 * pad)
    c = (int(16 + 14 * t), int(26 + 26 * t), int(40 + 34 * t), 255)
    bd.line([(pad, y), (S - pad, y)], fill=c)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([pad, pad, S - pad, S - pad], r, fill=255)
img.paste(bg, (0, 0), mask)

cx = cy = S / 2


def hexagon(radius, rot=0.0):
    return [(cx + radius * math.cos(math.radians(60 * i + rot)),
             cy + radius * math.sin(math.radians(60 * i + rot))) for i in range(6)]


# 外圈孔道
outer = hexagon(300, -90)
d.line(outer + [outer[0]], fill=(56, 189, 248, 255), width=30, joint="curve")

# 六个节点：外圈顶点画实心球
for i in range(6):
    x, y = outer[i]
    d.ellipse([x - 62, y - 62, x + 62, y + 62], fill=(56, 189, 248, 255))
    d.ellipse([x - 42, y - 42, x + 42, y + 42], fill=(240, 249, 255, 255))

# 中心：一个小的实心六边形，代表被围出来的孔
core = hexagon(96, -90)
d.polygon(core, fill=(34, 211, 238, 190))

img = img.resize((256, 256), Image.LANCZOS)
img.save("cof_icon.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
img.save("cof_icon.png")
print("图标已生成 cof_icon.ico")
