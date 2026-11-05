# -*- coding: utf-8 -*-
"""把 COF Studio 的源码和更新包发到 GitHub。

一条命令干完：提交源码 → 推上去 → 建 Release → 传三个资产
（latest.json / cof_patch_<版本>.zip / COF Studio <版本>.zip）。

准备：
    set GITHUB_TOKEN=ghp_xxxxxxxx          （Windows cmd）
    export GITHUB_TOKEN=ghp_xxxxxxxx       （Git Bash / PowerShell 用 $env:GITHUB_TOKEN）
Token 需要 repo 权限（classic）或 Contents + Administration（fine-grained）。

第一次（顺手把仓库建出来，公开）：
    python publish_github.py --repo 你的账号/COF-Studio --create-repo \
        --notes "首个公开版本：默认停在 2D 建模，界面中英切换，支持在线更新"

以后每次发版（先把 release/ 生成好，见 make_release.py）：
    python publish_github.py --repo 你的账号/COF-Studio --notes "修复 XXX"

只推源码不建 Release：
    python publish_github.py --repo 你的账号/COF-Studio --no-release
"""

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIO = HERE / "cof_studio.py"
RELEASE_DIR = HERE / "release"
API = "https://api.github.com"


def app_version() -> str:
    """从源码里抠版本号（不能 import，一 import 就把服务器拉起来了）。"""
    m = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"',
                  STUDIO.read_text(encoding="utf-8"), re.M)
    if not m:
        sys.exit("cof_studio.py 里找不到 APP_VERSION")
    return m.group(1)


def token() -> str:
    t = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not t:
        sys.exit("请先设置环境变量 GITHUB_TOKEN（需要 repo 权限）")
    return t


def api(method: str, path: str, body=None, ok=(200, 201), raw: bytes = None,
        ctype: str = "application/json", base: str = API):
    """调 GitHub REST。body 给 dict 就按 JSON 发，raw 给 bytes 就原样发。"""
    url = path if path.startswith("http") else base + path
    data = raw if raw is not None else (
        json.dumps(body).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + token(),
        "Accept": "application/vnd.github+json",
        "User-Agent": "COFStudio-publisher",
        "Content-Type": ctype,
    })
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            txt = r.read()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code in ok:
            return {"_status": e.code}
        raise RuntimeError(f"{method} {url} → {e.code}\n{detail}")


def git(*args, check=True):
    return subprocess.run(["git"] + list(args), cwd=str(HERE), check=check,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def ensure_repo_created(repo: str, desc: str) -> None:
    try:
        api("GET", f"/repos/{repo}")
        print(f"仓库已存在：{repo}")
        return
    except RuntimeError:
        pass
    name = repo.split("/")[-1]
    api("POST", "/user/repos", {
        "name": name, "private": False, "description": desc,
        "has_issues": True, "has_wiki": False, "auto_init": False,
    })
    print(f"已新建公开仓库：{repo}")


def push_source(repo: str, ver: str) -> None:
    """初始化（若需要）、提交、推送。Token 只在命令行里出现，不写进 .git/config。"""
    if not (HERE / ".git").is_dir():
        git("init", "-b", "main")
        print("已初始化 git 仓库")
    git("add", "-A")
    st = git("status", "--porcelain")
    if st.stdout.strip():
        git("-c", "user.name=COF Studio", "-c", "user.email=cof@localhost",
            "commit", "-m", f"COF Studio {ver}")
        print(f"已提交 {len(st.stdout.strip().splitlines())} 个变更")
    else:
        print("工作区没有变更，跳过提交")
    url = f"https://x-access-token:{token()}@github.com/{repo}.git"
    git("remote", "remove", "origin", check=False)
    git("remote", "add", "origin", url)
    try:
        r = git("push", "-u", "origin", "HEAD:main", check=False)
        if r.returncode != 0:
            raise RuntimeError("push 失败：\n" + (r.stderr or r.stdout))
    finally:
        # 别把带 token 的地址留在 .git/config 里
        git("remote", "set-url", "origin", f"https://github.com/{repo}.git",
            check=False)
    print(f"已推送源码到 {repo}")


def make_release(repo: str, ver: str, notes: str) -> dict:
    tag = "v" + ver
    try:
        old = api("GET", f"/repos/{repo}/releases/tags/{tag}")
        print(f"Release {tag} 已存在，先删掉重建")
        api("DELETE", f"/repos/{repo}/releases/{old['id']}", ok=(204,))
    except RuntimeError:
        pass
    rel = api("POST", f"/repos/{repo}/releases", {
        "tag_name": tag,
        "name": f"COF Studio {ver}",
        "body": notes or f"COF Studio {ver}",
        "draft": False,
        "prerelease": False,
    })
    print(f"已创建 Release {tag}")
    return rel


def upload_assets(repo: str, rel: dict, ver: str) -> None:
    """三个资产缺一不可：
         latest.json        客户端先读它，知道有哪些包、sha256 是多少
         cof_patch_<v>.zip  增量补丁（几百 KB，覆盖后刷新即生效）
         cof_full_<v>.zip   完整包（改了 exe 才需要，换目录重装）
    每个 Release 都要带齐，否则老版本用户拿不到整包。"""
    want = ["latest.json", f"cof_patch_{ver}.zip", f"cof_full_{ver}.zip"]
    missing = [n for n in want if not (RELEASE_DIR / n).is_file()]
    if missing:
        sys.exit("release/ 里缺文件（先跑 make_release.py）：\n  "
                 + "\n  ".join(missing))
    for name in want:
        p = RELEASE_DIR / name
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if name.endswith(".zip"):
            ctype = "application/zip"
        elif name.endswith(".json"):
            ctype = "application/json"
        data = p.read_bytes()
        url = (f"https://uploads.github.com/repos/{repo}/releases/"
               f"{rel['id']}/assets?name=" + urllib.parse.quote(name))
        api("POST", url, raw=data, ctype=ctype)
        print(f"  已上传 {name}（{len(data)/1024/1024:.1f} MB）")
    print(f"资产地址：https://github.com/{repo}/releases/tag/v{ver}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="账号/仓库名，如 rui/COF-Studio")
    ap.add_argument("--notes", default="", help="Release 说明")
    ap.add_argument("--create-repo", action="store_true", help="没有仓库就先建一个公开的")
    ap.add_argument("--no-release", action="store_true", help="只推源码，不建 Release")
    ap.add_argument("--desc", default="COF Studio：COF 结构建模与势函数优化（本地网页界面）")
    args = ap.parse_args()

    ver = app_version()
    print(f"版本 {ver} → {args.repo}\n")

    if args.create_repo:
        ensure_repo_created(args.repo, args.desc)
    push_source(args.repo, ver)
    if args.no_release:
        return
    rel = make_release(args.repo, ver, args.notes)
    upload_assets(args.repo, rel, ver)
    print("\n客户端「更新源」填：")
    print(f"  https://github.com/{args.repo}/releases/latest/download")


if __name__ == "__main__":
    main()
