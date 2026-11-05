#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COF Studio —— 建模 + 优化一条龙的本地工作台（零第三方 Web 依赖）

做什么
    python cof_studio.py   →  浏览器自动打开 http://127.0.0.1:8770
    ① 2D 页签：两个单体 SMILES → 二维 COF 单层，实时三维预览
    ② 3D 页签：两个单体 SMILES + 拓扑/连接方式 → 三维 COF 晶胞
    ③ 优化页签：把上面建出来的结构丢给 MACE 势弛豫，导出优化后的 CIF

为什么这样写
    - Web 部分只用标准库 http.server，不装 Flask/Dash —— 别人拿到就能跑；
    - 三维渲染是前端自己写的 canvas 投影，不走 CDN，断网也能用；
    - 真的要算的东西全部通过**子进程**调 engines/ 里的原脚本，
      界面不改动科学计算的任何一行代码，两边可以各自升级。

两套解释器
    建模要 numpy + ase + rdkit；优化要 numpy + ase + torch + mace。
    这两套依赖经常不在同一个环境里（torch 太重），所以 config.json 里
    可以给建模、优化分别指定解释器；留空就用启动本程序的解释器。

产物
    work/   下按体系名存档：<名>.cif、<名>_opt.cif、<名>_stage*.log …

启动
    python cof_studio.py                 # 默认 127.0.0.1:8770
    python cof_studio.py --port 9000 --open
    python cof_studio.py --no-browser    # 不自动开浏览器
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote, quote

FROZEN = bool(getattr(sys, "frozen", False))
HERE = Path(__file__).resolve().parent

if FROZEN:
    # 打包成 exe 之后路径要分成两份：
    #   RES  = 随包资源（ui.html / engines/），PyInstaller 解到临时目录
    #   DATA = exe 旁边那个目录，用户能看见、能写（work/、config.json、日志）
    RES = Path(getattr(sys, "_MEIPASS", HERE))
    DATA = Path(sys.executable).resolve().parent
else:
    RES = HERE
    DATA = HERE

CONFIG_FILE = DATA / "config.json"

# --------------------------------------------------------------------------
# 引导：主程序自己也要 numpy / ase / scipy
# --------------------------------------------------------------------------
# 常见坑：双击 run.bat 用的是系统自带 Python（什么包都没有），
# 但用户明明在另一个环境里装好了 —— 于是还没进界面就 ModuleNotFoundError。
# 这里先探一下，缺包就换到 config.json 里配的解释器重新执行自己。


def _has_deps(exe: str) -> bool:
    if not exe:
        return False
    if os.path.normcase(exe) == os.path.normcase(sys.executable):
        return False
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run(
            [exe, "-c", "import numpy, ase, scipy"],
            capture_output=True, timeout=60, env=env)
        return r.returncode == 0
    except Exception:
        return False


def _bootstrap() -> None:
    if FROZEN:
        return          # 依赖就在 exe 里，没得换
    try:
        import numpy    # noqa: F401
        import ase      # noqa: F401
        import scipy    # noqa: F401
        return
    except ImportError:
        pass

    # 候选：config 里配的 > .venv > 环境变量
    cands = []
    try:
        if CONFIG_FILE.is_file():
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            cands.append((cfg.get("py_build") or "").strip())
            cands.append((cfg.get("py_opt") or "").strip())
    except Exception:
        pass
    for p in (HERE / ".venv" / "Scripts" / "python.exe",
              HERE / ".venv" / "bin" / "python"):
        if p.is_file():
            cands.append(str(p))
    cands.append((os.environ.get("COF_PY") or "").strip())

    for exe in cands:
        if exe and Path(exe).is_file() and _has_deps(exe):
            print(f"[引导] 当前解释器缺 numpy/ase/scipy，改用 {exe} 重新启动…")
            sys.stdout.flush()
            rc = subprocess.call([exe, str(Path(__file__).resolve())]
                                 + sys.argv[1:])
            sys.exit(rc)

    sys.exit(
        "\n[缺少依赖] 这个 Python 里没有 numpy / ase / scipy：\n"
        f"    {sys.executable}\n\n"
        "  三个办法（任一）：\n"
        "    1. 双击 install.bat，自动建好环境并装齐依赖（推荐）\n"
        "    2. 用装了依赖的解释器启动：\n"
        "         <那个python.exe> cof_studio.py\n"
        "    3. 手动装：\n"
        "         pip install numpy ase scipy matplotlib rdkit\n")


_bootstrap()

import numpy as np                                     # noqa: E402
from ase.data import atomic_numbers, covalent_radii    # noqa: E402
from ase.io import read, write as ase_write            # noqa: E402

# --------------------------------------------------------------------------
# 路径与配置
# --------------------------------------------------------------------------
ENGINES = RES / "engines"
WORK = DATA / "work"
UI_FILE = RES / "ui.html"

WORK.mkdir(exist_ok=True)

# 引擎文件名（engines/ 里放的是随包的副本）
ENGINE_FILES = {
    "build_3d": "build_cof_3d.py",
    "build_2d": "cof_builder.py",
    "opt": "opt_cof.py",
}

DEFAULT_CONFIG = {
    # 引擎脚本路径；留空 = 用 engines/ 里的副本
    "engine_build_3d": "",
    "engine_build_2d": "",
    "engine_opt": "",
    # 解释器路径；留空 = 用启动本程序的解释器
    "py_build": "",
    "py_opt": "",
    # 界面语言：zh / en；留空 = 跟随浏览器
    # 存在这里而不是只放 localStorage，是因为 localStorage 按端口隔离，
    # 8770 被占用时后端会换端口，用户选的英文下次就丢了。
    "ui_lang": "",
    # 更新源目录：共享盘（\\NAS\COF\update）、本机路径或 http 地址
    # 预填发布者的更新源，装完不用手输地址就能"检查更新"。
    # 想换成局域网镜像，在 设置 → 版本与更新 里改（可填多个，换行分隔）。
    "update_url": "https://github.com/1783722112/COF-Studio/releases/latest/download",
}

APP_VERSION = "1.3.0"

# 这几个才是路径，load_config 的"路径还活着吗"自愈只该管它们。
PATH_KEYS = ("engine_build_3d", "engine_build_2d", "engine_opt",
             "py_build", "py_opt")

CFG_LOCK = threading.Lock()


def _path_alive(v: str) -> bool:
    """配置里的路径还活着吗。允许写裸命令名（如 'python'）。"""
    if not v:
        return True
    if os.path.isabs(v) or os.sep in v or "/" in v:
        return Path(v).is_file()
    return shutil.which(v) is not None


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    # 自愈：config 里可能是别人机器上的绝对路径（把整个目录打包发出去的情况）
    changed = False
    for k in PATH_KEYS:
        v = (cfg.get(k) or "").strip()
        if v and not _path_alive(v):
            print(f"[配置] {k} 指向的路径不存在，已回落到默认：{v}")
            cfg[k] = ""
            changed = True
    if changed:
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    with CFG_LOCK:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


CFG = load_config()


def engine_path(key: str) -> Path:
    """配置里指定了、且文件确实在，就用它；否则回落到 engines/ 副本。"""
    p = (CFG.get("engine_" + key) or "").strip()
    if p:
        cand = Path(p).expanduser()
        if cand.is_file():
            return cand
    return ENGINES / ENGINE_FILES[key]


def _venv_py() -> str:
    """install.bat 建出来的 .venv 优先用 —— 别人装完不用再改配置。"""
    for p in (HERE / ".venv" / "Scripts" / "python.exe",
              HERE / ".venv" / "bin" / "python"):
        if p.is_file():
            return str(p)
    return ""


def build_py() -> str:
    return (CFG.get("py_build") or "").strip() or _venv_py() or sys.executable


def opt_py() -> str:
    return (CFG.get("py_opt") or "").strip() or build_py()


# --------------------------------------------------------------------------
# 版本与更新
#
# 为什么分成"补丁"和"整包"两种：
#   ui.html 和 engines/*.py 在 onedir 包里是明文数据文件（不是编译进 exe 的），
#   所以改了界面或建模引擎，只要覆盖这两个地方就行 —— 几百 KB、刷新即生效。
#   只有 cof_studio.py 本身（接口、启动逻辑）动过，才必须换 exe，那才走整包。
# --------------------------------------------------------------------------
UPDATE_DIR = DATA / "update"


def current_version() -> str:
    """当前版本。热更新会写一份 version.txt，优先级高于 exe 里编译进去的号 ——
    否则补丁装完了版本号还停在旧值，每次检查都提示"有新版本"。"""
    vf = RES / "version.txt"
    if vf.is_file():
        try:
            v = vf.read_text(encoding="utf-8").strip()
            if v:
                return v
        except Exception:
            pass
    return APP_VERSION


def _ver_tuple(v) -> tuple:
    """'1.2.3' -> (1, 2, 3)。解析不了的段按 0 算，别让脏版本号把检查搞崩。"""
    out = []
    for part in str(v or "").strip().lstrip("vV").split("."):
        num = ""
        for ch in part:
            if not ch.isdigit():
                break
            num += ch
        out.append(int(num) if num else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


# 最近一次成功读到 latest.json 的源。多源时后面的补丁/整包下载都走这一个，
# 不要每次重新猜（局域网和 GitHub 混着填的时候，猜错就白下一遍）。
_ACTIVE_BASE = ""


def _update_bases() -> list:
    """update_url 可以填多个：换行或分号分隔，按顺序优先。"""
    raw = (CFG.get("update_url") or "").strip()
    out = []
    for part in re.split(r"[;\r\n]+", raw):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out


def _update_base() -> str:
    if _ACTIVE_BASE:
        return _ACTIVE_BASE
    bases = _update_bases()
    return bases[0] if bases else ""


def _join_url(base: str, name: str) -> str:
    name = str(name or "").strip()
    # latest.json 里的 path 允许直接写完整 URL（比如补丁放 GitHub Release、
    # 整包放公司网盘），这时不再拼 base。
    if name.lower().startswith(("http://", "https://", "file://")):
        return name
    # 包名里出现空格/中文要编码，否则 urllib 直接抛 InvalidURL。
    # （自己做 release 时已经用 cof_patch_/cof_full_ 这种无空格名字，这里只是兜底）
    if re.search(r"[^\x21-\x7e]|[\s\"'<>#]", name):
        name = quote(name, safe="/")
    return base.replace("\\", "/").rstrip("/") + "/" + name


def _fetch_bytes(url: str, limit: int = 400 * 1024 * 1024) -> bytes:
    """读更新包里的文件。支持 http(s)、file://、UNC 共享盘和普通盘符路径。

    局域网共享目录是最常见的用法（\\\\NAS\\COF\\update），所以不能只认 http。
    """
    u = url.strip()
    if u.lower().startswith(("http://", "https://")):
        import urllib.request
        # 先按系统默认（可能走用户的代理——国内访问 GitHub 往往必须走），
        # 失败再禁掉代理直连试一次。两条路都不通才算失败。
        # 注意重试必须新建 Request：第一次失败会在 req 上残留 _tunnel_host，
        # 复用对象会让"禁代理"的 opener 仍然走代理隧道。
        try:
            req = urllib.request.Request(
                u, headers={"User-Agent": "COFStudio/" + current_version()})
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read(limit + 1)
        except Exception:
            req = urllib.request.Request(
                u, headers={"User-Agent": "COFStudio/" + current_version()})
            direct = urllib.request.build_opener(
                urllib.request.ProxyHandler({}))   # 空 handler = 禁代理
            with direct.open(req, timeout=120) as r:
                blob = r.read(limit + 1)
    else:
        if u.lower().startswith("file://"):
            u = unquote(urlparse(u).path)
            if re.match(r"^/[A-Za-z]:", u):     # file:///C:/... → C:/...
                u = u[1:]
        with open(u, "rb") as f:
            blob = f.read(limit + 1)
    if len(blob) > limit:
        raise RuntimeError("文件超过预期大小，已中止")
    return blob


def _sha256(blob: bytes) -> str:
    import hashlib
    return hashlib.sha256(blob).hexdigest()


def read_update_index() -> dict:
    global _ACTIVE_BASE
    bases = _update_bases()
    if not bases:
        raise RuntimeError("还没填更新源（设置 → 版本与更新）")
    errs = []
    for base in bases:
        try:
            raw = _fetch_bytes(_join_url(base, "latest.json"), 4 * 1024 * 1024)
            idx = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(idx, dict) or not idx.get("version"):
                raise RuntimeError("latest.json 格式不对：缺 version 字段")
            _ACTIVE_BASE = base
            return idx
        except Exception as e:                  # 换下一个源
            errs.append(f"{base} → {type(e).__name__}: {e}")
    raise RuntimeError("所有更新源都取不到 latest.json：\n  " + "\n  ".join(errs))


# 补丁只许碰这三类文件。zip 是外部给的，白名单比"事后检查"可靠。
PATCH_ALLOW = ("ui.html", "engines/", "version.txt")


def _safe_extract(zf, dest: Path, allow=None) -> int:
    """解压到 dest，挡住 ../ 和绝对路径。

    一个被换过的更新包能往任意位置写文件 —— 解压前必须把路径洗干净，
    光靠"来源可信"不够。
    """
    dest = dest.resolve()
    n = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/").lstrip("/")
        if ".." in name.split("/") or re.match(r"^[A-Za-z]:", name):
            raise RuntimeError(f"更新包里有非法路径：{info.filename}")
        if allow is not None and not any(
                name == p.rstrip("/") or name.startswith(p) for p in allow):
            raise RuntimeError(f"更新包里有不该出现的文件：{info.filename}")
        target = (dest / name).resolve()
        if dest not in target.parents and target != dest:
            raise RuntimeError(f"更新包里有非法路径：{info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        n += 1
    return n


def check_update() -> dict:
    idx = read_update_index()
    latest = str(idx.get("version"))
    patch = idx.get("patch") or {}
    full = idx.get("full") or {}
    return {
        "ok": True,
        "current": current_version(),
        "latest": latest,
        "newer": _ver_tuple(latest) > _ver_tuple(current_version()),
        "date": idx.get("date", ""),
        "notes": idx.get("notes", ""),
        "has_patch": bool(patch.get("path")),
        "has_full": bool(full.get("path")),
        "source": _update_base(),
        "frozen": FROZEN,
        "can_apply": FROZEN,          # 源码模式不让它自己改文件
    }


def apply_patch(idx: dict) -> dict:
    """增量更新：只覆盖 ui.html / engines/ / version.txt。"""
    if not FROZEN:
        raise RuntimeError("源码模式请直接改文件，不走在线更新")
    asset = idx.get("patch") or {}
    if not asset.get("path"):
        raise RuntimeError("这个版本没有提供补丁包，只能用整包更新")
    url = asset["url"] if asset.get("url") else _join_url(_update_base(), asset["path"])
    blob = _fetch_bytes(url)
    want = (asset.get("sha256") or "").lower()
    got = _sha256(blob)
    if want and got != want:
        raise RuntimeError(f"补丁校验失败，可能没传完（sha256 {got[:8]} ≠ {want[:8]}）")
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        if not [i for i in zf.infolist() if not i.is_dir()]:
            raise RuntimeError("补丁包是空的")
        tmp = UPDATE_DIR / ("patch_" + str(idx.get("version") or "new"))
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        count = _safe_extract(zf, tmp, PATCH_ALLOW)
        # 先备份将被覆盖的文件，装坏了还能拿回来
        bak = RES / "_backup" / ("before_" + current_version())
        replaced = []
        for src in sorted(tmp.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(tmp)
            dst = RES / rel
            if dst.is_file():
                b = bak / rel
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, b)
                replaced.append(rel.as_posix())
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # 补丁里如果没有 version.txt，自己写一份，免得版本号对不上
    if not (tmp / "version.txt").is_file():
        (RES / "version.txt").write_text(str(idx.get("version")), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True, "files": count, "replaced": replaced,
            "backup": str(bak), "version": current_version(),
            "need_restart": True}


def stage_full(idx: dict) -> dict:
    """整包更新：下载 → 解压到旁边的新目录 → 等用户重启时换上去。"""
    if not FROZEN:
        raise RuntimeError("源码模式不支持整包更新")
    asset = idx.get("full") or {}
    if not asset.get("path"):
        raise RuntimeError("这个版本没有提供整包")
    url = asset["url"] if asset.get("url") else _join_url(_update_base(), asset["path"])
    blob = _fetch_bytes(url)
    want = (asset.get("sha256") or "").lower()
    got = _sha256(blob)
    if want and got != want:
        raise RuntimeError(f"整包校验失败，可能没传完（sha256 {got[:8]} ≠ {want[:8]}）")

    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    zpath = UPDATE_DIR / ("full_" + str(idx.get("version") or "new") + ".zip")
    zpath.write_bytes(blob)

    newdir = DATA.parent / (DATA.name + ".new")
    if newdir.exists():
        shutil.rmtree(newdir, ignore_errors=True)
    newdir.mkdir(parents=True, exist_ok=True)
    import zipfile
    with zipfile.ZipFile(zpath) as zf:
        _safe_extract(zf, newdir)          # 解到空目录，路径穿越已经挡掉
        # 包里可能裹了一层 "COF Studio/"，拆掉
        if not (newdir / "COFStudio.exe").is_file():
            subs = [d for d in newdir.iterdir() if d.is_dir()]
            if len(subs) == 1 and (subs[0] / "COFStudio.exe").is_file():
                for item in list(subs[0].iterdir()):
                    shutil.move(str(item), str(newdir / item.name))
                subs[0].rmdir()
    if not (newdir / "COFStudio.exe").is_file():
        raise RuntimeError("整包里没找到 COFStudio.exe")
    zpath.unlink(missing_ok=True)
    return {"ok": True, "dir": str(newdir), "bytes": len(blob),
            "version": str(idx.get("version"))}


def _cleanup_old_versions(keep: int = 1) -> None:
    """清掉更新留下的旧版本目录，只留最近一份用于回滚。

    更新器只做改名（删 170 MB 会被杀毒/索引拖住几十秒，实测整包更新会卡死），
    所以备份会攒下来；这里在程序正常运行时异步清，删不掉就算，不影响启动。
    """
    try:
        olds = [p for p in DATA.parent.glob("_old_" + DATA.name + "*") if p.is_dir()]
        olds.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in olds[keep:]:
            shutil.rmtree(p, ignore_errors=True)
            print(f"[清理] 已删除旧版本目录：{p.name}")
    except Exception:
        pass


def install_staged() -> dict:
    """把 ".new" 目录换成正式的。

    正在运行的 exe 锁着自己 —— Windows 不允许删/改一个正在执行的镜像，
    所以只能交给一个独立进程：它等我们退出，再改名、重启。

    为什么用 PowerShell 而不是批处理：
      ① 批处理里 `tasklist | find` 会踩到 PATH —— 机器上装了 Git/MSYS 时
         find 指向 GNU find，语法不对，等待循环永远出不来（实测卡死）；
      ② 批处理进程的 cwd 默认就是 COF Studio，而 Windows 不允许重命名
         当前工作目录，move 必然失败。改用 .NET 的 Directory.Move 之前
         必须先把 cwd 挪走，脚本里第一件事就是 Set-Location。
    """
    if not FROZEN:
        raise RuntimeError("源码模式不跑安装器")
    newdir = DATA.parent / (DATA.name + ".new")
    if not (newdir / "COFStudio.exe").is_file():
        raise RuntimeError("新版还没解压好，先点「下载整包」")

    def q(p) -> str:
        return "'" + str(p).replace("'", "''") + "'"

    ps = "\n".join([
        # 不能用 Stop：下面要"试着重命名、失败就重试"，Stop 会让第一次重试
        # 直接终结整个脚本。真正的失败由 try/catch 兜。
        "$ErrorActionPreference = 'Continue'",
        f"$root = {q(DATA.parent)}",
        f"$cur  = {q(DATA.name)}",
        f"$new  = {q(newdir.name)}",
        # 逐步留痕：装失败时不用猜是哪一步 —— 看 %TEMP%\\cof_update_trace.log
        "$trace = Join-Path $env:TEMP 'cof_update_trace.log'",
        "function T([string]$m) { try { Add-Content -LiteralPath $trace -Encoding UTF8 "
        "-Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $m) } catch {} }",
        # 这几个放 try 外面：万一中途抛错，最后还要靠 $src 把程序拉起来
        "$src = Join-Path $root $cur",
        "$dst = Join-Path $root $new",
        "$old = Join-Path $root ('_old_' + $cur)",
        "$ack = Join-Path $env:TEMP 'cof_install_alive.txt'",
        "try {",
        "  T 'start'",
        "  try { Set-Content -LiteralPath $ack -Value 'ok' -Encoding UTF8 } catch {}",
        # 不能待在要被改名的目录里（Windows 不允许重命名当前工作目录）
        "  Set-Location -LiteralPath $env:TEMP",
        "  T ('cwd=' + (Get-Location).Path)",
        # 主程序在 2 秒后自己退出；这里多等 1 秒，剩下的交给重试。
        # 刻意不用 Get-Process 等 PID：PID 会被系统复用，等错了进程会卡死。
        "  Start-Sleep -Seconds 3",
        "  T 'slept'",
        # 上一份备份不动它：这里**绝不删除**旧目录。
        # 实测删 170 MB 的旧版会被杀毒/索引服务拖住几十秒，整个更新就卡死。
        # 只做改名，秒级完成；旧备份交给程序下次启动时清理。
        "  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'",
        "  $old = Join-Path $root ('_old_' + $cur + '_' + $stamp)",
        "  if (Test-Path -LiteralPath $old) { $old = $old + '.' + (Get-Random) }",
        # 刚退出的进程可能还持有句柄，重试到目录真正放开为止
        "  $moved = $false",
        "  for ($i = 0; $i -lt 20; $i++) {",
        "    try { [System.IO.Directory]::Move($src, $old); $moved = $true; break }",
        "    catch { Start-Sleep -Milliseconds 500 }",
        "  }",
        "  T ('current -> backup: ' + $moved)",
        "  if (-not $moved) { throw 'rename current dir failed (still locked)' }",
        "  try {",
        "    [System.IO.Directory]::Move($dst, $src)",
        "    T 'new -> current'",
        "  } catch {",
        # 换新失败就把旧的改回来，别让用户面对一个不存在的目录
        "    T ('new move failed: ' + $_.Exception.Message)",
        "    [System.IO.Directory]::Move($old, $src)",
        "    throw",
        "  }",
        "} catch {",
        "  T ('FAILED: ' + $_.Exception.Message)",
        "  Add-Content -LiteralPath (Join-Path $root 'update_failed.log') -Encoding UTF8 "
        "-Value ('[' + (Get-Date -Format s) + '] ' + $_.Exception.Message)",
        "}",
        "Start-Process -FilePath (Join-Path $src 'COFStudio.exe') -WorkingDirectory $src",
    ]) + "\n"

    ps1 = Path(os.environ.get("TEMP") or str(DATA.parent)) / "cof_install_update.ps1"
    ps1.write_text(ps, encoding="utf-8-sig")     # 5.1 要 BOM 才不当成本地编码

    exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" \
        / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    runner = str(exe) if exe.is_file() else "powershell"
    # 以「脱离当前进程」的方式拉起安装脚本。
    # 安装器要等我们退出才能给目录改名，所以它必须比我们活得久：
    #   · DETACHED_PROCESS 让它不继承我们的控制台/进程组；
    #   · Windows 默认不会因为父进程退出而杀掉子进程（除非宿主用了
    #     KILL_ON_JOB_CLOSE 的 job 对象 —— 沙箱和 CI 里常见，正常桌面没有）。
    # 万一 Popen 起不来，退回 os.startfile（走文件关联），最差也能让用户看到脚本。
    flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
             | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        subprocess.Popen(
            [runner, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-WindowStyle", "Hidden", "-File", str(ps1)],
            cwd=str(ps1.parent), close_fds=True, creationflags=flags,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[更新] 拉起安装脚本失败（{e}），改用文件关联")
        os.startfile(str(ps1))

    # 等安装脚本回执再退出：它第一步就写这个文件。这样能确认"它真的起来了"，
    # 而不是我们退了它却已经被系统回收 —— 那种情况下用户会看到一个凭空消失的程序。
    ack = Path(os.environ.get("TEMP") or str(DATA.parent)) / "cof_install_alive.txt"
    try:
        ack.unlink()
    except Exception:
        pass
    for _ in range(80):                     # 最多等 8 秒
        if ack.is_file():
            break
        time.sleep(0.1)
    else:
        print("[更新] 安装脚本没有回执，可能被安全软件拦下")

    # 给前端留时间把"正在重启"渲染出来，然后自己退出，把目录让出去
    threading.Timer(2.0, lambda: os._exit(0)).start()
    return {"ok": True, "restarting": True}


# --------------------------------------------------------------------------
# 引擎调用：打包成 exe 之后不能再指望外部解释器
# --------------------------------------------------------------------------
# 源码模式下是  "python.exe  engines/xxx.py  <参数>"。
# 打包之后没有那个 python.exe 了，于是让 exe 自己再执行一遍自己：
#     "COFStudio.exe --pyexec  engines/xxx.py  <参数>"
# --pyexec 分支在 main() 最前面，不走界面、不弹窗口，纯当一个解释器用。
def _self_exec(argv: list) -> list:
    return [sys.executable, "--pyexec"] + list(argv)


def xrd_cmd(args: list) -> list:
    """XRD 引擎跟建模引擎共用解释器（只要 numpy + ase）。"""
    script = str(ENGINES / "xrd.py")
    if FROZEN:
        return _self_exec([script] + list(args))
    return [build_py(), script] + list(args)


def engine_cmd(key: str, args: list) -> list:
    """拼出调用某个引擎的完整命令行。"""
    script = str(engine_path(key))
    args = list(args)
    if FROZEN:
        if key == "opt":
            # 优化要 torch + MACE（约 2 GB），没随包打进去。
            # 用户若在 config.json 里指了外部解释器，就用他的。
            exe = (CFG.get("py_opt") or "").strip()
            if exe and app_exists(exe):
                return [exe, script] + args
        return _self_exec([script] + args)
    py = opt_py() if key == "opt" else build_py()
    return [py, script] + args


# --------------------------------------------------------------------------
# 解释器自检
# --------------------------------------------------------------------------
MODULES = ["numpy", "ase", "rdkit", "torch", "mace", "matplotlib", "scipy"]

_PROBE_SRC = r"""
import json, sys, importlib
out = {"python": sys.version.split()[0], "exe": sys.executable, "packages": {}}
for m in %s:
    try:
        mod = importlib.import_module(m)
        out["packages"][m] = getattr(mod, "__version__", "ok")
    except Exception:
        out["packages"][m] = None
try:
    import torch
    out["cuda"] = bool(torch.cuda.is_available())
    out["device"] = torch.cuda.get_device_name(0) if out["cuda"] else "CPU only"
except Exception:
    out["cuda"] = False
    out["device"] = "-"
print("<<<PROBE>>>" + json.dumps(out))
""" % (MODULES,)

_probe_cache: dict[str, tuple[float, dict]] = {}


def probe_python(exe: str, ttl: float = 900.0) -> dict:
    """跑一段自检脚本问这个解释器装了哪些包。结果缓存 15 分钟。"""
    exe = exe or sys.executable
    now = time.time()
    hit = _probe_cache.get(exe)
    if hit and now - hit[0] < ttl:
        return hit[1]

    info = {"exe": exe, "ok": False, "python": "?", "packages": {},
            "cuda": False, "device": "-", "error": ""}
    if not app_exists(exe):
        info["error"] = "解释器不存在"
        _probe_cache[exe] = (now, info)
        return info

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        if FROZEN and os.path.normcase(exe) == os.path.normcase(sys.executable):
            # 问自己：exe --pyexec -c "..."
            probe = _self_exec(["-c", _PROBE_SRC])
        else:
            probe = [exe, "-c", _PROBE_SRC]
        p = subprocess.run(probe, capture_output=True,
                           text=True, timeout=90, encoding="utf-8",
                           errors="replace", env=env, cwd=str(HERE))
        for line in (p.stdout or "").splitlines():
            if line.startswith("<<<PROBE>>>"):
                info.update(json.loads(line[11:]))
                info["ok"] = True
                break
        else:
            err = (p.stderr or p.stdout or "").strip()
            info["error"] = err[-300:] or f"退出码 {p.returncode}"
    except subprocess.TimeoutExpired:
        info["error"] = "自检超时（90s）"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    _probe_cache[exe] = (now, info)
    return info


def app_exists(exe: str) -> bool:
    if not exe:
        return False
    if os.path.isabs(exe) or os.sep in exe or "/" in exe:
        return Path(exe).is_file()
    return shutil.which(exe) is not None


def env_report() -> dict:
    """给界面的环境面板：建模链、优化链各自缺什么。"""
    b = probe_python(build_py())
    o = probe_python(opt_py())

    def lack(info, need):
        return [m for m in need if not (info.get("packages") or {}).get(m)]

    build_need = ["numpy", "ase", "rdkit"]
    opt_need = ["numpy", "ase", "torch", "mace"]

    rep = {
        "build": {
            "exe": b["exe"], "python": b["python"], "ok": b["ok"],
            "packages": b["packages"], "missing": lack(b, build_need),
            "error": b.get("error", ""),
        },
        "opt": {
            "exe": o["exe"], "python": o["python"], "ok": o["ok"],
            "packages": o["packages"], "missing": lack(o, opt_need),
            "error": o.get("error", ""), "cuda": o.get("cuda", False),
            "device": o.get("device", "-"),
        },
        "same_interp": os.path.normcase(b["exe"]) == os.path.normcase(o["exe"]),
        "engines": {k: str(engine_path(k)) for k in ENGINE_FILES},
        "engines_ok": {k: engine_path(k).is_file() for k in ENGINE_FILES},
    }
    rep["can_build"] = rep["build"]["ok"] and not rep["build"]["missing"]
    rep["can_opt"] = rep["opt"]["ok"] and not rep["opt"]["missing"]
    return rep


# --------------------------------------------------------------------------
# 结构 → 前端用的 JSON
# --------------------------------------------------------------------------
ELEM_COLORS = {
    "H": "#f2f4f8", "B": "#ffb5b5", "C": "#6d7889", "N": "#4a7ff5",
    "O": "#f0524a", "F": "#7ee0a0", "S": "#e8c547", "P": "#ff9e4a",
    "Cl": "#5edc7f", "Br": "#b4632c", "I": "#a463d6", "Si": "#c9a06a",
    "Zn": "#7d80b0", "Cu": "#c88033", "Fe": "#e06633", "Ni": "#50d050",
}


def _bonds_with_ghosts(pos, cell, syms, cutoff_pad=1.28, max_atoms=2600):
    """算键。跨胞边界的键要把镜像原子补出来（ghost），否则边上的骨架会断。

    返回 (atoms, bonds, n_unit)：atoms 前 n_unit 个是胞内原子，之后是 ghost。
    """
    n = len(pos)
    R = np.array([covalent_radii[atomic_numbers[s]] for s in syms])
    out_pos = [tuple(map(float, p)) for p in pos]
    ghost_of = {}                     # (unit_idx, shift_tuple) -> 新索引
    bonds, seen = [], set()

    shifts = [(i, j, k) for i in (-1, 0, 1)
              for j in (-1, 0, 1) for k in (-1, 0, 1)]
    shifts_cart = np.array(shifts, float) @ np.array(cell, float)

    from scipy.spatial import cKDTree
    allpos = (pos[None, :, :] + shifts_cart[:, None, :]).reshape(-1, 3)
    owner = np.tile(np.arange(n), len(shifts))
    tree = cKDTree(allpos)

    dmax = float(R.max() * 2 * cutoff_pad) if n else 2.0
    for i in range(n):
        for jj in tree.query_ball_point(pos[i], dmax):
            k = int(owner[jj])
            if k == i:
                continue
            d = float(np.linalg.norm(allpos[jj] - pos[i]))
            if d < 0.4 or d > cutoff_pad * (R[i] + R[k]):
                continue
            key = (min(i, k), max(i, k))
            if key in seen:
                continue
            seen.add(key)
            # 判断这条键是否落在本胞内：镜像位移为 0 且另一端也在胞内
            sh = shifts[jj // n]
            if sh == (0, 0, 0):
                bonds.append([i, k])
            else:
                gk = (k, sh)
                if gk not in ghost_of and len(out_pos) < max_atoms:
                    ghost_of[gk] = len(out_pos)
                    out_pos.append(tuple(map(float, allpos[jj])))
                if gk in ghost_of:
                    bonds.append([i, ghost_of[gk]])
    syms_all = list(syms) + [syms[k] for (k, _) in ghost_of]
    return syms_all, out_pos, bonds, n, len(ghost_of)


def structure_payload(path: Path, with_ghost=True, tile=None,
                      auto_tile=True) -> dict:
    """读一个 CIF/XYZ，产出前端渲染需要的全部数据 + 体检指标。

    tile=(nx, ny) 时把胞平铺再算键 —— 二维 COF 只画一个胞的话，
    孔洞大多落在角上，屏幕上只看得到"半个孔"，看不出是蜂窝格子。
    auto_tile 会自动给薄层结构做 2×2。
    """
    a = read(str(path))
    syms0 = a.get_chemical_symbols()
    pos0 = a.get_positions()
    cell = np.array(a.cell)
    n = len(a)

    # ---- 自动决定要不要平铺：薄层（原子在 z 方向展布远小于 xy）才铺 ----
    if auto_tile and tile is None and n <= 420:
        # 注意 numpy 2.0 起 ndarray.ptp() 被移除，必须用 np.ptp(arr)
        dxp = float(np.ptp(pos0[:, 0])) if n else 0.0
        dyp = float(np.ptp(pos0[:, 1])) if n else 0.0
        dzp = float(np.ptp(pos0[:, 2])) if n else 0.0
        if dzp < 0.35 * max(dxp, dyp, 1e-6):
            tile = (2, 2)
    if tile and (tile[0] > 1 or tile[1] > 1):
        nx, ny = int(tile[0]), int(tile[1])
        sc = cell.copy()
        sc[0] = cell[0] * nx
        sc[1] = cell[1] * ny
        blocks, bsym = [], []
        for i in range(nx):
            for j in range(ny):
                blocks.append(pos0 + i * cell[0] + j * cell[1])
                bsym.extend(syms0)
        posT = np.vstack(blocks)
        symsT = bsym
        syms_all, pos_all, bonds, n_unit, n_ghost = _bonds_with_ghosts(
            posT, sc, symsT)
        cell_out = sc
        n_tile = (nx, ny)
    else:
        n_tile = (1, 1)
        cell_out = cell
        if with_ghost and n <= 2600:
            syms_all, pos_all, bonds, n_unit, n_ghost = _bonds_with_ghosts(
                pos0, cell, syms0)
        else:
            syms_all, pos_all, n_ghost = syms0, [tuple(p) for p in pos0], 0
            bonds, n_unit = [], n
            if n <= 6000:
                from scipy.spatial import cKDTree
                R = np.array([covalent_radii[atomic_numbers[s]] for s in syms0])
                tree = cKDTree(pos0)
                for i in range(n):
                    for k in tree.query_ball_point(pos0[i],
                                                   float(R.max()) * 2.6):
                        if k <= i:
                            continue
                        d = float(np.linalg.norm(pos0[k] - pos0[i]))
                        if 0.4 < d < 1.28 * (R[i] + R[k]):
                            bonds.append([i, int(k)])

    # -------- 体检 --------
    d = a.get_all_distances(mic=True)
    np.fill_diagonal(d, 99.0)
    dmin = float(d.min()) if n > 1 else 0.0
    cp = a.cell.cellpar()          # 注意：cell 已是 ndarray，cellpar 要问 ase 的 Cell
    vol = float(a.get_volume())
    try:
        mass = float(a.get_masses().sum())
        density = mass / 6.02214076e23 / (vol * 1e-24) if vol > 1 else 0.0
    except Exception:
        density = 0.0

    # 连接键：C–N / C–O 在 1.2–1.45 之间
    link = None
    try:
        het = [i for i, s in enumerate(syms0) if s in ("N", "O")]
        cand = []
        for i in het:
            for j in range(n):
                if j == i or syms0[j] != "C":
                    continue
                dij = float(d[i, j])
                if 1.15 <= dij <= 1.45:
                    cand.append((i, j, dij))
        if cand:
            vals = np.array([c[2] for c in cand])
            link = {"n": len(cand), "mean": float(vals.mean()),
                    "std": float(vals.std())}
    except Exception:
        pass

    return {
        "ok": True,
        "name": path.stem,
        "atoms": [[s, round(p[0], 4), round(p[1], 4), round(p[2], 4)]
                  for s, p in zip(syms_all, pos_all)],
        "bonds": bonds,
        # cell 给前端画晶胞框，平铺时是超胞；cellpar 永远是**原胞**参数
        "cell": [[round(float(v), 4) for v in row] for row in cell_out],
        "tile": list(n_tile),
        "n_unit": n_unit,
        "n_ghost": n_ghost,
        "formula": a.get_chemical_formula(),
        "cellpar": [round(float(x), 3) for x in cp],
        "volume": round(vol, 2),
        "density": round(density, 4),
        "dmin": round(dmin, 3),
        "linkage": link,
        "colors": ELEM_COLORS,
    }


# --------------------------------------------------------------------------
# 建模（同步，几秒）
# --------------------------------------------------------------------------
def _clean_smiles(s: str) -> str:
    """把用户从网页/Word 里粘来的空白、全角字符、首尾引号清掉。"""
    if not s:
        return ""
    s = s.replace("\u3000", " ").replace("\u00a0", " ")
    s = s.strip().strip("'\"`")
    s = re.sub(r"\s+", "", s)
    return s


def _run_sync(cmd, cwd, timeout=600):
    env = dict(os.environ,
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace",
                       env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def safe_name(name: str, default="structure") -> str:
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    name = name.strip("._") or default
    return name[:60]


def build_3d(params: dict) -> dict:
    """调 build_cof_3d.py。参数同它自己的 CLI。"""
    node = _clean_smiles(params.get("node"))
    linker = _clean_smiles(params.get("linker"))
    if not node or not linker:
        return {"ok": False, "error": "两个单体的 SMILES 都要填"}

    topo = (params.get("topology") or "ctn").strip()
    linkage = (params.get("linkage") or "imine").strip()
    cell = str(params.get("cell") or "auto").strip() or "auto"
    name = safe_name(params.get("name") or f"3D_{topo}")

    out = WORK / name
    cmd = engine_cmd("build_3d",
                     ["-n", node, "-l", linker,
                      "--topology", topo, "--linkage", linkage,
                      "-a", cell, "-o", str(out)])
    t0 = time.time()
    try:
        rc, log = _run_sync(cmd, WORK, timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "建模超时（>15 分钟）", "log": ""}

    cif = out.with_suffix(".cif")
    if rc != 0 or not cif.exists():
        tail = "\n".join(log.strip().splitlines()[-25:])
        return {"ok": False, "error": "建模失败，见下方日志", "log": tail}

    payload = structure_payload(cif)
    payload.update({"log": "\n".join(log.strip().splitlines()[-30:]),
                    "elapsed": round(time.time() - t0, 2),
                    "file": cif.name, "kind": "3d"})
    return payload


def _stack_layers(cif: Path, mode: str, layers: int, gap: float,
                  dx: float, dy: float) -> Path:
    """把单层结构沿 c 轴复制成 N 层，另存一个 CIF 并返回路径。

    AA   → 每层完全对齐（eclipsed），c = gap × N
    AB   → 奇偶层交替偏移 (dx, dy)（默认 0.5, 0.5），交错堆积
    自由 → 每层比上一层再多偏 (dx, dy)，连续滑移（ABC 型）

    偏移量是**分数坐标**，直接加在 x / y 上，不依赖晶胞形状。
    """
    from ase import Atoms
    from ase.io import read as _rd, write as _wr

    a = _rd(str(cif))
    cell = np.array(a.cell, dtype=float)
    frac = np.array(a.get_scaled_positions(wrap=True), dtype=float)
    syms = a.get_chemical_symbols()
    if not len(syms):
        raise RuntimeError("结构里没有原子")
    if abs(cell[2][2]) < 1e-6:
        raise RuntimeError("c 轴长度为 0，没法堆叠")

    new_cell = cell.copy()
    new_cell[2] = cell[2] * layers
    af, asy = [], []
    for i in range(layers):
        if mode == "ab":
            ox, oy = ((dx, dy) if i % 2 else (0.0, 0.0))
        elif mode == "free":
            ox, oy = dx * i, dy * i
        else:
            ox, oy = 0.0, 0.0
        f = frac.copy()
        f[:, 0] += ox
        f[:, 1] += oy
        f[:, 2] = (f[:, 2] + i) / layers
        af.append(f)
        asy.extend(syms)

    new = Atoms(symbols=asy, scaled_positions=np.vstack(af),
                cell=new_cell, pbc=True)
    tag = {"aa": "AA", "ab": "AB"}.get(mode, "free")
    outp = cif.with_name(f"{cif.stem}_{tag}{layers}L.cif")
    _wr(str(outp), new, format="cif")
    return outp


def build_2d(params: dict) -> dict:
    node = _clean_smiles(params.get("node"))
    linker = _clean_smiles(params.get("linker"))
    if not node or not linker:
        return {"ok": False, "error": "两个单体的 SMILES 都要填"}

    linkage = (params.get("linkage") or "imine").strip()
    topo = (params.get("topology") or "auto").strip()
    name = safe_name(params.get("name") or f"2D_{linkage}")

    # ---- 堆叠参数：AA（默认）/ AB / 自由滑移 ----
    stack = (params.get("stack") or "aa").strip().lower()
    if stack not in ("aa", "ab", "free"):
        stack = "aa"
    try:
        layers = int(params.get("layers") or 1)
    except Exception:
        layers = 1
    layers = max(1, min(layers, 6))
    try:
        gap = float(params.get("gap") or 3.60)
    except Exception:
        gap = 3.60
    gap = max(2.5, min(gap, 8.0))
    try:
        dx = float(params.get("dx") or 0.0)
        dy = float(params.get("dy") or 0.0)
    except Exception:
        dx = dy = 0.0
    if stack == "aa":
        dx = dy = 0.0
        if layers == 1:
            layers = 2              # 默认 AA 双层：能看出层间峰，又不至于太慢
    elif layers < 2:
        layers = 2                  # 选了 AB / 自由，没两层没意义

    out = WORK / name
    args = ["-n", node, "-l", linker, "-k", linkage,
            "-o", str(out), "--no-plot", "--c", f"{gap:.4f}"]
    if topo and topo != "auto":
        args += ["--topology", topo]
    cmd = engine_cmd("build_2d", args)

    t0 = time.time()
    try:
        rc, log = _run_sync(cmd, engine_path("build_2d").parent, timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "建模超时（>15 分钟）", "log": ""}

    cif = out.with_suffix(".cif")
    if rc != 0 or not cif.exists():
        tail = "\n".join(log.strip().splitlines()[-25:])
        return {"ok": False, "error": "建模失败，见下方日志", "log": tail}

    target, stacked = cif, ""
    if layers > 1:
        try:
            target = _stack_layers(cif, stack, layers, gap, dx, dy)
            stacked = target.name
        except Exception as e:
            return {"ok": False,
                    "error": f"堆叠失败：{type(e).__name__}: {e}",
                    "log": "\n".join(log.strip().splitlines()[-20:])}

    payload = structure_payload(target)
    payload.update({"log": "\n".join(log.strip().splitlines()[-30:]),
                    "elapsed": round(time.time() - t0, 2),
                    "file": target.name, "kind": "2d",
                    "single": cif.name if stacked else "",
                    "stack": {"mode": stack, "layers": layers, "gap": gap,
                              "dx": dx, "dy": dy}})
    if stacked:
        payload["log"] = (f"[堆叠] {stack.upper()} × {layers} 层，层间距 "
                          f"{gap:.2f} Å，偏移 ({dx:.3f}, {dy:.3f}) → {stacked}\n"
                          + payload["log"])
    return payload


# --------------------------------------------------------------------------
# XRD 模拟
# --------------------------------------------------------------------------
def run_xrd(params: dict) -> dict:
    """按给定 2θ 范围算粉末 XRD。结构取 work/ 里的 CIF/XYZ。"""
    p = _safe_work_file(params.get("file") or "")
    if not p:
        return {"ok": False,
                "error": "找不到结构文件（先建模，或从 work/ 里挑一个）"}
    if p.suffix.lower() not in (".cif", ".xyz"):
        return {"ok": False, "error": "只支持 CIF / XYZ"}

    def num(key, dflt, lo, hi):
        try:
            v = float(params.get(key))
        except Exception:
            v = dflt
        return max(lo, min(v, hi))

    wl = num("wl", 1.54056, 0.3, 3.5)
    t0 = num("tth_min", 2.0, 0.5, 120.0)
    t1 = num("tth_max", 40.0, 1.0, 150.0)
    if t1 <= t0 + 0.5:
        return {"ok": False, "error": "2θ 范围要至少 0.5°，且终点大于起点"}
    step = num("step", 0.02, 0.002, 0.5)
    if (t1 - t0) / step > 20000:
        return {"ok": False, "error": "步长太小（超过 2 万个点），调大一点"}
    size = num("size", 20.0, 0.0, 5000.0)

    cmd = xrd_cmd(["--cif", str(p), "--wl", f"{wl:.5f}",
                   "--tth-min", f"{t0:.3f}", "--tth-max", f"{t1:.3f}",
                   "--step", f"{step:.4f}", "--size", f"{size:.1f}"])
    t_start = time.time()
    try:
        rc, log = _run_sync(cmd, ENGINES, timeout=600)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "XRD 计算超时（>10 分钟）"}

    data = None
    for line in reversed((log or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                break
            except Exception:
                continue
    if not data:
        return {"ok": False, "error": "XRD 引擎没有返回结果",
                "log": "\n".join((log or "").strip().splitlines()[-15:])}
    if not data.get("ok"):
        return {"ok": False, "error": data.get("error") or "XRD 计算出错"}

    csv_name = ""
    try:
        csv_name = f"{p.stem}_xrd_{t0:.0f}-{t1:.0f}.csv"
        with open(WORK / csv_name, "w", encoding="utf-8") as f:
            f.write("2theta_deg,intensity\n")
            for x, v in zip(data["tth"], data["I"]):
                f.write(f"{x:.3f},{v:.2f}\n")
    except Exception:
        csv_name = ""

    data.update({"ok": True, "file": p.name, "csv": csv_name,
                 "elapsed": round(time.time() - t_start, 2),
                 "size": size})
    return data


# --------------------------------------------------------------------------
# 优化任务（异步）
# --------------------------------------------------------------------------
class Job:
    def __init__(self, jid: str, cmd: list, cwd: Path, out_cif: Path,
                 label: str, meta: dict | None = None):
        self.id = jid
        self.cmd = cmd
        self.cwd = cwd
        self.out_cif = out_cif
        self.label = label
        self.meta = meta or {}
        self.lines: deque = deque(maxlen=4000)
        self.status = "running"      # running / done / failed / cancelled
        self.rc = None
        self.t0 = time.time()
        self.t1 = None
        self.stage = 0
        self.step = 0
        self.proc = None
        self._stop = False
        self._last_scan = 0.0
        self._th = threading.Thread(target=self._run, daemon=True)

    # ---------- 生命周期 ----------
    def start(self):
        self._th.start()

    def stop(self):
        self._stop = True
        p = self.proc
        if p and p.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   capture_output=True)
                else:
                    p.terminate()
            except Exception:
                pass
        self._push("[已请求停止]")

    def _push(self, line: str):
        self.lines.append(line)

    def _run(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                self.cmd, cwd=str(self.cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1, encoding="utf-8",
                errors="replace", env=env, **kw)
        except Exception as exc:
            self.status = "failed"
            self._push(f"[启动失败] {type(exc).__name__}: {exc}")
            self.t1 = time.time()
            return

        self._push("$ " + " ".join(self.cmd))
        try:
            for raw in iter(self.proc.stdout.readline, ""):
                raw = raw.rstrip("\n").rstrip("\r")
                if not raw:
                    continue
                if "\r" in raw:                 # 进度条就地刷新
                    raw = raw.split("\r")[-1]
                if not raw.strip():
                    continue
                self._push(raw)
                self._sniff(raw)
                if self._stop:
                    break
        except Exception as exc:
            self._push(f"[读取输出出错] {exc}")

        try:
            self.rc = self.proc.wait(timeout=30)
        except Exception:
            self.rc = None
        self.t1 = time.time()

        if self._stop:
            self.status = "cancelled"
        elif self.out_cif.exists() and (self.rc in (0, None)):
            self.status = "done"
            self._push(f"[完成] 产物 {self.out_cif.name}")
        elif self.out_cif.exists():
            # 脚本偶尔在最后一步非零退出但文件已写好
            self.status = "done"
            self._push(f"[完成] 产物 {self.out_cif.name}（退出码 {self.rc}）")
        else:
            self.status = "failed"
            self._push(f"[失败] 退出码 {self.rc}，没有生成 {self.out_cif.name}")

    def _sniff(self, line: str):
        m = re.search(r">>>\s*阶段\s*(\d)", line)
        if m:
            self.stage = int(m.group(1))
            self.step = 0
        m = re.search(r"完成\s*(\d+)\s*步", line)
        if m:
            self.step = int(m.group(1))

    def _live_step(self):
        """ase 的 FIRE/BFGS 把每一行写进 logfile，stdout 只在阶段结束才说话。
        不读这个文件的话，整个阶段 1 进度条都停在 0，看着像卡死。"""
        now = time.time()
        if now - self._last_scan < 1.5:
            return
        self._last_scan = now
        try:
            logs = sorted(self.cwd.glob(f"{self.out_cif.stem}_stage*.log"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return
        if not logs:
            return
        cur = logs[0]
        m = re.search(r"stage(\d)", cur.name)
        if m:
            self.stage = max(self.stage, int(m.group(1)))
        try:
            with open(cur, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4000))
                tail = fh.read().decode("utf-8", "replace")
        except Exception:
            return
        for line in reversed(tail.splitlines()):
            mm = re.match(r"\s*(?:FIRE|BFGS|LBFGS|MD)\s*:\s*(\d+)", line)
            if mm:
                try:
                    self.step = int(mm.group(1))
                except ValueError:
                    pass
                return

    # ---------- 状态 ----------
    def snapshot(self, start: int = 0) -> dict:
        if self.status == "running":
            self._live_step()
        lines = list(self.lines)
        return {
            "id": self.id, "label": self.label, "status": self.status,
            "stage": self.stage, "step": self.step,
            "elapsed": round((self.t1 or time.time()) - self.t0, 1),
            "lines": lines[start:], "total": len(lines),
            "rc": self.rc,
            "out": self.out_cif.name if self.out_cif.exists() else None,
            "cmd": " ".join(self.cmd),
            **self.meta,
        }


JOBS: dict[str, Job] = {}
JOB_SEQ = [0]


def guess_dimension(cellpar) -> str:
    """猜这是 3D 框架还是 2D 单层。

    opt_cof.py 是按二维 COF 写的：它沿层法向找空隙，觉得"层间距太小"就
    补 15 Å 真空再算，输出时再把真空抽掉。对 2D 单层这是对的，对三维框架
    就是灾难 —— a=b=c=24.677 的立方胞会被当成"层厚 14.7 + 间隙 0.2"，
    输出时压成 c=21 甚至 18 Å，整块骨架被切断。

    判据只有一条：三维框架的三个轴尺度相当（c 跟 a 差不到 15%），
    二维 COF 的 c 是层间距（3.5 Å 或补了真空的 20+ Å），永远偏。
    判错的代价很小：带真空的 2D 即便被判成 3D，也只是"保留原胞"，
    不会改动几何。真正的开关在界面上，用户可以自己改。
    """
    try:
        a, b, c = [float(x) for x in cellpar[:3]]
    except Exception:
        return "2d"
    if a < 1e-6:
        return "2d"
    return "3d" if abs(c - a) < 0.15 * a and abs(c - b) < 0.15 * a else "2d"


def start_opt(params: dict) -> dict:
    cif_name = (params.get("file") or "").strip()
    if not cif_name:
        return {"ok": False, "error": "先建一个结构，或者从产物里挑一个"}
    src = (WORK / safe_name(cif_name)).resolve()
    if not src.is_file():
        # 允许带扩展名 / 不带扩展名
        for cand in (WORK / safe_name(cif_name + ".cif"), WORK / cif_name):
            if cand.is_file():
                src = cand.resolve()
                break
        else:
            return {"ok": False, "error": f"找不到 {cif_name}"}

    pot = (params.get("potential") or "qcof").strip()
    if pot not in ("qcof", "mace-off", "mace-mp", "file"):
        pot = "qcof"

    # ---- 结构维度：决定要不要让脚本"补真空 / 抽真空" ----
    dim = (params.get("dimension") or "auto").strip().lower()
    if dim not in ("auto", "2d", "3d"):
        dim = "auto"
    cellpar = None
    try:
        _a = read(str(src))
        cellpar = _a.cell.cellpar()
    except Exception:
        pass
    if dim == "auto":
        dim = guess_dimension(cellpar) if cellpar is not None else "2d"

    out_name = safe_name(params.get("out") or (src.stem + "_opt"))
    out = WORK / (out_name + ".cif")

    if FROZEN and not (CFG.get("py_opt") or "").strip():
        return {"ok": False, "error": (
            "这个桌面版不含势函数环境 —— torch + MACE 权重约 2 GB，没有打包进去。\n\n"
            "本机若另有装了 torch / mace-torch 的 Python，在 exe 旁边的 "
            "config.json 里把 py_opt 指向那个 python.exe 就能用了。")}

    cmd = engine_cmd("opt", [str(src), "-o", str(out), "-p", pot, "--no-plot"])

    if dim == "3d":
        # 三维框架的两个关键开关：
        #   --no-vacuum    别往里塞 15 Å 真空（会把骨架沿 c 切开）
        #   --keep-vacuum  输出也别抽（否则按 3.45 Å 堆垛间距把 c 压扁）
        cmd += ["--no-vacuum", "--keep-vacuum"]

    model = (params.get("model") or "").strip()
    if model in ("small", "medium", "large"):
        cmd += ["--model", model]
    if params.get("model_file"):
        cmd += ["--model-file", str(params["model_file"])]
    if params.get("device") in ("cpu", "cuda", "auto"):
        cmd += ["--device", params["device"]]
    for key, flag in (("relax_steps", "--relax-steps"),
                      ("cell_steps", "--cell-steps"),
                      ("relax_fmax", "--relax-fmax"),
                      ("cell_fmax", "--cell-fmax")):
        v = params.get(key)
        if v not in (None, ""):
            cmd += [flag, str(v)]
    if params.get("no_cell"):
        cmd += ["--no-cell"]
    if params.get("fix_c"):
        cmd += ["--fix-c"]
    if params.get("no_dispersion"):
        cmd += ["--no-dispersion"]
    if params.get("keep_vacuum"):
        cmd += ["--keep-vacuum"]
    if params.get("allow_download"):
        cmd += ["--allow-download"]

    JOB_SEQ[0] += 1
    jid = f"j{JOB_SEQ[0]}"
    meta = {
        "potential": pot,
        "dimension": dim,
        "relax_steps": int(params.get("relax_steps") or 200),
        "cell_steps": 0 if params.get("no_cell") else int(
            params.get("cell_steps") or 500),
        "no_cell": bool(params.get("no_cell")),
    }
    job = Job(jid, cmd, WORK, out, label=src.stem, meta=meta)
    # 只保留最近 20 个任务，免得内存里堆一堆日志
    if len(JOBS) > 20:
        for k in [k for k, j in JOBS.items() if j.status != "running"][:-10]:
            JOBS.pop(k, None)
    JOBS[jid] = job
    job.start()
    return {"ok": True, "job": jid, "out": out.name,
            "cmd": " ".join(cmd), "dimension": dim}


# --------------------------------------------------------------------------
# 文件列表
# --------------------------------------------------------------------------
def list_files() -> list:
    out = []
    for p in sorted(WORK.glob("*.cif"), key=lambda x: -x.stat().st_mtime)[:60]:
        st = p.stat()
        out.append({"name": p.name, "size": st.st_size,
                    "mtime": time.strftime("%m-%d %H:%M",
                                           time.localtime(st.st_mtime)),
                    "opt": p.stem.endswith("_opt")})
    return out


def _safe_work_file(name: str) -> Path | None:
    name = unquote(name or "").strip().replace("\\", "/").split("/")[-1]
    if not name:
        return None
    p = (WORK / name).resolve()
    try:
        p.relative_to(WORK.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "COFStudio/2.0"

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # ---------- 工具 ----------
    def _send(self, code, body: bytes, ctype: str, extra=None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 浏览器提前关了连接（刷新页面、切页签）属正常，不该在日志里刷堆栈
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # ---------- GET ----------
    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                if not UI_FILE.is_file():
                    return self._send(500, b"ui.html missing",
                                      "text/plain; charset=utf-8")
                # 把记住的界面语言注入页面，省掉"先按浏览器语言画一遍、
                # 再异步纠正"的闪烁。占位符不存在（旧版 ui.html）就原样返回。
                html = UI_FILE.read_text(encoding="utf-8")
                if "<!--UI_LANG-->" in html:
                    _lang = (CFG.get("ui_lang") or "").strip()
                    html = html.replace(
                        "<!--UI_LANG-->",
                        "<script>window.__UI_LANG__=%s;</script>"
                        % json.dumps(_lang if _lang in ("zh", "en") else ""))
                return self._send(200, html.encode("utf-8"),
                                  "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path == "/api/env":
                if q.get("force"):
                    _probe_cache.clear()
                return self._json(env_report())
            if path == "/api/files":
                return self._json({"files": list_files()})
            if path == "/api/version":
                # 上次整包更新失败会留下 update_failed.log；读一次就删，
                # 免得每次开都弹同一个提示。
                _fail = DATA / "update_failed.log"
                _failed = _fail.is_file()
                if _failed:
                    try:
                        _fail.unlink()
                    except Exception:
                        pass
                return self._json({"version": current_version(),
                                   "built": APP_VERSION,
                                   "frozen": FROZEN,
                                   "update_failed": _failed,
                                   "update_url": CFG.get("update_url") or ""})
            if path == "/api/config":
                return self._json({"config": CFG,
                                   "engines": {k: str(engine_path(k))
                                               for k in ENGINE_FILES}})
            if path == "/api/job":
                jid = (q.get("id") or [""])[0]
                job = JOBS.get(jid)
                if not job:
                    return self._json({"ok": False, "error": "任务不存在"}, 404)
                start = int((q.get("from") or ["0"])[0])
                return self._json({"ok": True, **job.snapshot(start)})
            if path == "/api/jobs":
                # 页面刷新后靠这个把正在跑的任务接回来，不然日志就断在半路了
                return self._json({"jobs": [
                    {"id": j.id, "label": j.label, "status": j.status,
                     "elapsed": round((j.t1 or time.time()) - j.t0, 1),
                     "out": j.out_cif.name}
                    for j in JOBS.values()]})
            if path == "/api/struct":
                p = _safe_work_file((q.get("f") or [""])[0])
                if not p:
                    return self._json({"ok": False, "error": "文件不存在"}, 404)
                return self._json(structure_payload(p))
            if path == "/api/download":
                p = _safe_work_file((q.get("f") or [""])[0])
                if not p:
                    return self._send(404, b"not found", "text/plain")
                data = p.read_bytes()
                return self._send(
                    200, data, "application/octet-stream",
                    {"Content-Disposition":
                     f'attachment; filename="{p.name}"'})
            return self._send(404, b"not found", "text/plain")
        except Exception:
            traceback.print_exc()
            return self._json({"ok": False,
                               "error": traceback.format_exc()[-800:]}, 500)

    # ---------- POST ----------
    def do_POST(self):
        u = urlparse(self.path)
        path = u.path
        try:
            req = self._body()
            if path == "/api/build":
                mode = (req.get("mode") or "3d").lower()
                res = build_2d(req) if mode == "2d" else build_3d(req)
                return self._json(res)
            if path == "/api/xrd":
                return self._json(run_xrd(req))
            if path == "/api/opt":
                return self._json(start_opt(req))
            if path == "/api/job/stop":
                job = JOBS.get(req.get("id") or "")
                if job:
                    job.stop()
                    return self._json({"ok": True})
                return self._json({"ok": False, "error": "任务不存在"}, 404)
            if path == "/api/update/check":
                try:
                    return self._json(check_update())
                except Exception as e:
                    return self._json({"ok": False,
                                       "error": f"{type(e).__name__}: {e}"})
            if path == "/api/update/apply":
                mode = (req.get("mode") or "patch").strip()
                try:
                    if mode == "check":          # 一步到位：检查 + 装补丁
                        idx = read_update_index()
                        if not (_ver_tuple(str(idx.get("version"))) >
                                _ver_tuple(current_version())):
                            return self._json({"ok": False,
                                               "error": "已经是最新版本"})
                        return self._json(apply_patch(idx))
                    if mode == "patch":
                        return self._json(apply_patch(read_update_index()))
                    if mode == "stage":          # 只下载整包，先不换
                        return self._json(stage_full(read_update_index()))
                    if mode == "install":        # 换目录并重启
                        return self._json(install_staged())
                    return self._json({"ok": False, "error": f"未知更新模式 {mode}"}, 400)
                except Exception as e:
                    return self._json({"ok": False,
                                       "error": f"{type(e).__name__}: {e}"})
            if path == "/api/config":
                for k in list(DEFAULT_CONFIG):
                    if k not in req:
                        continue
                    v = str(req[k] or "").strip()
                    if k == "ui_lang" and v not in ("", "zh", "en"):
                        continue
                    CFG[k] = v
                save_config(CFG)
                _probe_cache.clear()
                return self._json({"ok": True, "config": CFG})
            if path == "/api/reveal":            # 在资源管理器里打开 work/
                try:
                    if os.name == "nt":
                        os.startfile(str(WORK))       # noqa: S606
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(WORK)])
                    else:
                        subprocess.Popen(["xdg-open", str(WORK)])
                    return self._json({"ok": True})
                except Exception as exc:
                    return self._json({"ok": False, "error": str(exc)})
            return self._send(404, b"not found", "text/plain")
        except Exception:
            traceback.print_exc()
            return self._json({"ok": False,
                               "error": traceback.format_exc()[-800:]}, 500)


# --------------------------------------------------------------------------
# 启动
# --------------------------------------------------------------------------
BANNER = r"""
   ____ ___  _____    ____  _             _ _
  / ___/ _ \|  ___|  / ___|| |_ _   _  __| (_) ___
 | |  | | | | |_     \___ \| __| | | |/ _` | |/ _ \
 | |__| |_| |  _|     ___) | |_| |_| | (_| | | (_) |
  \____\___/|_|      |____/ \__|\__,_|\__,_|_|\___/

  COF Studio  ·  建模 + 势函数优化
"""


def find_free_port(host: str, start: int, tries: int = 30) -> int:
    import socket
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start


def _force_utf8_io() -> None:
    """把 stdout/stderr 钉成 UTF-8。

    窗口模式的 exe 没有控制台，子进程的 stdout 默认按系统代码页（简体中文 = gbk）编码，
    而引擎日志里到处是 "Å"、"∠"、"①" 这种 gbk 编不出来的字符 —— 一 print 就
    UnicodeEncodeError，进程静悄悄死掉，父进程只看到"建模失败"却没有任何日志。
    """
    for name, fd in (("stdout", 1), ("stderr", 2)):
        s = getattr(sys, name, None)
        try:
            if s is None:
                s = open(fd, "w", encoding="utf-8", errors="replace",
                         buffering=1, closefd=False)
                setattr(sys, name, s)
            else:
                s.reconfigure(encoding="utf-8", errors="replace",
                              line_buffering=True)
        except Exception:
            pass


def _pyexec(argv: list) -> int:
    """把本进程当解释器用：执行一段代码，或者跑一个脚本。

        COFStudio.exe --pyexec engines/xxx.py <它自己的参数…>
        COFStudio.exe --pyexec -c "<代码>"

    打包成 exe 之后没有外部 python.exe 了，建模引擎就走这条路进来。
    """
    _force_utf8_io()
    if not argv:
        return 2
    if argv[0] == "-c":
        exec(compile(argv[1], "<probe>", "exec"), {"__name__": "__main__"})
        return 0

    script = Path(argv[0]).resolve()
    if not script.is_file():
        print(f"[引擎] 找不到脚本：{script}")
        return 2
    sys.argv = [str(script)] + list(argv[1:])
    # 引擎之间会互相 import（cof_builder → build_cof_smiles），脚本目录必须在 sys.path
    for d in (str(script.parent), str(Path.cwd())):
        if d not in sys.path:
            sys.path.insert(0, d)
    import runpy
    import traceback as _tb
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        # 引擎崩了也要把原因写进日志，不然界面上只有一个"失败"
        _tb.print_exc()
        sys.stderr.flush()
        return 1
    return 0


def _msgbox(text: str, title: str = "COF Studio", icon: int = 0x10) -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, icon)
    except Exception:
        pass


def _hide_console() -> None:
    try:
        import ctypes
        h = ctypes.windll.kernel32.GetConsoleWindow()
        if h:
            ctypes.windll.user32.ShowWindow(h, 0)          # SW_HIDE
    except Exception:
        pass


def _log_to_file() -> None:
    """窗口模式下没有控制台，输出写进 exe 旁边的日志，出事好查。"""
    try:
        f = open(DATA / "COFStudio.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====\n")
    except Exception:
        pass


def _open_app_window(url: str) -> bool:
    """用 Edge/Chrome 的应用模式开一个没有地址栏的窗口 —— 看起来就是个独立软件。"""
    if os.name != "nt":
        return False
    cands = []
    for env in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if not base:
            continue
        cands.append(Path(base) / "Microsoft/Edge/Application/msedge.exe")
        cands.append(Path(base) / "Google/Chrome/Application/chrome.exe")
    for exe in cands:
        if exe.is_file():
            try:
                subprocess.Popen([str(exe), f"--app={url}",
                                  "--window-size=1460,940", "--no-first-run"],
                                 creationflags=getattr(subprocess,
                                                       "CREATE_NO_WINDOW", 0))
                return True
            except Exception:
                continue
    return False


def _already_running(url: str) -> bool:
    """已经开着一个了？是的话直接把窗口调出来，别再起第二个服务。"""
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/env", timeout=1.5) as r:
            return b"can_build" in r.read(4096)
    except Exception:
        return False


def main():
    # 打包后的 exe 会把自己当解释器再拉起来跑引擎 —— 这条分支必须排在解析参数之前
    if len(sys.argv) > 2 and sys.argv[1] == "--pyexec":
        sys.exit(_pyexec(sys.argv[2:]))

    if hasattr(sys.stdout, "reconfigure"):
        try:
            # line_buffering：被 run.bat / 管道捕获时也能马上看到启动信息
            sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                                   line_buffering=True)
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="COF Studio —— 建模 + 优化一条龙的本地工作台")
    ap.add_argument("--port", type=int, default=8770, help="端口（默认 8770）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址（默认只给本机用）")
    ap.add_argument("--open", action="store_true", help="启动后自动开浏览器")
    ap.add_argument("--no-browser", action="store_true", help="不自动开浏览器")
    ap.add_argument("--verbose", action="store_true", help="打印每条 HTTP 请求")
    args = ap.parse_args()

    if FROZEN:
        _hide_console()
        _log_to_file()

    if not UI_FILE.is_file():
        msg = f"找不到界面文件：{UI_FILE}\n\n安装包不完整，请重新完整解压。"
        if FROZEN:
            _msgbox(msg, "COF Studio 启动失败")
        sys.exit(msg)
    for k, fn in ENGINE_FILES.items():
        if not engine_path(k).is_file():
            print(f"[注意] 引擎缺失：{engine_path(k)}")

    # 双击第二次时不要起两个服务，直接把已有窗口叫出来
    if FROZEN and _already_running(f"http://{args.host}:{args.port}"):
        url0 = f"http://{args.host}:{args.port}"
        print(f"已有实例在 {url0} 运行，打开窗口。")
        if not (_open_app_window(url0) or webbrowser.open(url0)):
            _msgbox(f"已有实例在运行：{url0}", "COF Studio", 0x40)
        return

    port = find_free_port(args.host, args.port)
    srv = ThreadingHTTPServer((args.host, port), Handler)
    srv.daemon_threads = True
    srv.verbose = args.verbose

    url = f"http://{args.host}:{port}"
    print(BANNER)
    print(f"  界面地址   {url}")
    print(f"  产物目录   {WORK}")
    print(f"  建模解释器 {'（exe 内置）' if FROZEN else build_py()}")
    print(f"  优化解释器 {opt_py()}")
    if port != args.port:
        print(f"  （{args.port} 被占用，自动换到 {port}）")
    print("\n  按 Ctrl+C 停止。\n")

    # 整包更新只改名不删（删 170 MB 会被杀毒拖住），垃圾会攒下来。
    # 放到后台做，删不动就算了，不挡启动。
    threading.Thread(target=_cleanup_old_versions, daemon=True).start()

    def _launch():
        # 先试"应用窗口"（无地址栏），失败再退回默认浏览器
        if not _open_app_window(url):
            webbrowser.open(url)

    if not args.no_browser:
        threading.Timer(0.8, _launch).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        for j in JOBS.values():
            j.stop()
        srv.server_close()


if __name__ == "__main__":
    # 被自己当解释器拉起来的那个进程（--pyexec）不弹窗、不写错误日志，
    # 出错就把堆栈吐到 stderr，由父进程收进建模日志里给用户看。
    if len(sys.argv) > 1 and sys.argv[1] == "--pyexec":
        try:
            sys.exit(main())
        except SystemExit:
            raise
        except BaseException:
            _force_utf8_io()
            traceback.print_exc()
            sys.exit(1)

    try:
        main()
    except SystemExit:
        raise
    except Exception:
        _tb = traceback.format_exc()
        try:
            (DATA / "COFStudio_error.log").write_text(_tb, encoding="utf-8")
        except Exception:
            pass
        if FROZEN:
            _msgbox("启动失败：\n\n" + _tb[-1200:], "COF Studio 启动失败")
        else:
            raise
