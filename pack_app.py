# -*- coding: utf-8 -*-
"""把 dist 出来的 "COF Studio" 文件夹压成一个可分享的 zip。

用法（在 COF_Studio 目录下）：
    python pack_app.py
产物：桌面/COF Studio 桌面版.zip

注意：中文文件名要显式标 UTF-8 标志位，否则对方用 WinRAR / 7-Zip 解压会乱码。
"""
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "COF Studio"
OUT = HERE.parent / "COF Studio 桌面版.zip"

if not SRC.is_dir():
    sys.exit(f"找不到 {SRC}，先跑 build_exe.bat 打包。")

files = []
# 运行产物不进包：config.json 里是本机解释器路径和界面语言，work/ 是用户建的
# 结构，日志也是本机跑出来的。带出去会让别人一打开就是别人的配置。
SKIP_DIRS = {"work", "__pycache__", ".build_tmp"}
SKIP_FILES = {"config.json", "COFStudio.log", "cof_studio.log"}
for root, dirs, names in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for n in names:
        if n in SKIP_FILES:
            continue
        p = Path(root) / n
        files.append(p)
print(f"打包 {len(files)} 个文件 -> {OUT.name}")

t0 = time.time()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for i, p in enumerate(files, 1):
        arc = Path("COF Studio") / p.relative_to(SRC)
        z.write(p, arc.as_posix())
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} …")
            sys.stdout.flush()

mb = OUT.stat().st_size / 1024 / 1024
print(f"完成：{OUT}  {mb:.0f} MB  用时 {time.time() - t0:.0f}s")
