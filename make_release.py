# -*- coding: utf-8 -*-
"""打一个「更新源」目录：latest.json + 补丁包 +（可选）整包。

用法（在 COF_Studio 目录下）：

    python make_release.py --notes "默认停在 2D 建模；界面语言可记住"
    python make_release.py --notes "..." --full "..\\COF Studio 桌面版.zip"

产物在 release\\ 下。把这个目录整个拷到共享盘（例如 \\\\NAS\\COF\\update），
然后在软件的「设置 → 版本与更新 → 更新源」里填这个路径即可。

两级包的区别：
  补丁  ui.html + engines/*.py + version.txt，几百 KB，覆盖后刷新页面就生效；
  整包  PyInstaller 出来的完整目录 zip（~70 MB），只有改了 cof_studio.py
        （接口、启动逻辑）才需要，装的时候要退出进程换目录。
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINES = HERE / "engines"
UI = HERE / "ui.html"
STUDIO = HERE / "cof_studio.py"

# 补丁里带哪些文件。必须和 cof_studio.py 的 PATCH_ALLOW 对得上，
# 否则装的时候会被白名单挡回来。
PATCH_FILES = [("ui.html", UI), ("version.txt", None)]
PATCH_FILES += [(f"engines/{p.name}", p) for p in sorted(ENGINES.glob("*.py"))]


def app_version() -> str:
    """从源码里抠版本号。

    不能 import cof_studio —— 它在模块级会检查依赖、缺包就 re-exec 自己，
    一 import 就把服务器拉起来了。
    """
    m = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', STUDIO.read_text(encoding="utf-8"),
                  re.M)
    if not m:
        sys.exit("cof_studio.py 里找不到 APP_VERSION")
    return m.group(1)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="", help="更新说明（显示给用户看）")
    ap.add_argument("--full", default="", help="完整包 zip 路径（可选）")
    ap.add_argument("--out", default="release", help="输出目录，默认 release/")
    ap.add_argument("--no-patch", action="store_true", help="只发整包，不打补丁")
    args = ap.parse_args()

    ver = app_version()
    out = (HERE / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = {
        "version": ver,
        "date": time.strftime("%Y-%m-%d"),
        "notes": args.notes or f"COF Studio {ver}",
    }

    if not args.no_patch:
        missing = [str(src) for _, src in PATCH_FILES if src is not None and not src.is_file()]
        if missing:
            sys.exit("这些文件不存在，没法打补丁：\n  " + "\n  ".join(missing))
        zpath = out / f"cof_patch_{ver}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, src in PATCH_FILES:
                if src is None:                       # version.txt 现写
                    z.writestr(name, ver)
                else:
                    z.write(src, name)
        index["patch"] = {
            "path": zpath.name,
            "sha256": sha256_of(zpath),
            "size": zpath.stat().st_size,
        }
        print(f"补丁  {zpath.name}  {zpath.stat().st_size/1024:.0f} KB  "
              f"{len(PATCH_FILES)} 个文件")

    if args.full:
        src = Path(args.full).expanduser()
        if not src.is_file():
            sys.exit(f"找不到整包：{src}")
        # 资产名要能直接拼进 URL：GitHub 的下载地址里带空格会 400
        dst = out / f"cof_full_{ver}.zip"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        index["full"] = {
            "path": dst.name,
            "sha256": sha256_of(dst),
            "size": dst.stat().st_size,
        }
        print(f"整包  {dst.name}  {dst.stat().st_size/1024/1024:.0f} MB")

    (out / "latest.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n版本 {ver} → {out}")
    print("把这个目录整个拷到共享盘，用户端填：「设置 → 版本与更新 → 更新源」")


if __name__ == "__main__":
    main()
