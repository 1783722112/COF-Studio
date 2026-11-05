"""把 COF Studio 打成一个可以直接发给别人的便携压缩包。

用法：
    python make_share.py            # 需要先跑过：runtime/ 里已装好依赖

生成的包里有 Python 运行时（runtime/）和全部依赖，对方解压后双击 run.bat 即可，
不需要装 Python、不需要联网。

打包时会做三件事：
  1. config.json 换成空配置（避免把本机的 Python 绝对路径带给别人）
  2. work/ 只放两个示例 CIF，不带本机的工作产物
  3. 剔除 __pycache__ / *.pyc / pip 缓存
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = "COF_Studio"
OUT = SRC.parent / "COF_Studio_便携版.zip"

FILES = [
    "cof_studio.py",
    "ui.html",
    "run.bat",
    "install.bat",
    "install.sh",
    "run.sh",
    "README.md",
    "预览_3D建模.png",
    "预览_2D建模.png",
    "预览_优化页.png",
    "预览_腙建模.png",
]
DIRS = ["engines", "runtime", "examples"]
SAMPLES = ["BF-COF-1.cif", "LZU-1.cif"]
SKIP_DIRS = {"__pycache__", ".git", ".idea", ".vscode", ".ipynb_checkpoints"}
SKIP_EXT = {".pyc", ".pyo"}


def main() -> None:
    if not (SRC / "runtime" / "python.exe").exists():
        sys.exit("找不到 runtime/python.exe —— 这个目录不是便携版，没法打成免安装包。\n"
                 "先准备 runtime（见 README），或者只发源码让对方自己 install.bat。")

    total = 0
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in FILES:
            p = SRC / f
            if p.is_file():
                z.write(p, f"{ROOT}/{f}")
                total += p.stat().st_size
                n += 1

        for d in DIRS:
            base = SRC / d
            if not base.is_dir():
                continue
            for root, dirs, files in os.walk(base):
                dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
                for fn in files:
                    p = Path(root) / fn
                    if p.suffix.lower() in SKIP_EXT:
                        continue
                    z.write(p, f"{ROOT}/{p.relative_to(SRC)}")
                    total += p.stat().st_size
                    n += 1

        # 空配置：对方机器上残留的绝对路径会让自检先报一次错，干脆不带
        z.writestr(f"{ROOT}/config.json", json.dumps({
            "engine_build_3d": "", "engine_build_2d": "", "engine_opt": "",
            "py_build": "", "py_opt": "",
        }, indent=2) + "\n")

        for s in SAMPLES:
            p = SRC / "work" / s
            if p.is_file():
                z.write(p, f"{ROOT}/work/{s}")
                n += 1

    size = OUT.stat().st_size
    print(f"打包完成：{OUT}")
    print(f"  {n} 个文件，原始 {total / 1e6:.0f} MB → 压缩后 {size / 1e6:.0f} MB")
    print(f"\n对方拿到后：完整解压 → 双击 {ROOT}/run.bat")


if __name__ == "__main__":
    main()
