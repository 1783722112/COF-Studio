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
        # 先按系统默认（可能走用户代理），失败再禁代理直连——和客户端
        # _fetch_bytes 同一套思路，国内网络两头都得备着
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                txt = r.read()
        except Exception:
            direct = urllib.request.build_opener(
                urllib.request.ProxyHandler({}))
            with direct.open(req, timeout=600) as r:
                txt = r.read()
        return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code in ok:
            return {"_status": e.code}
        raise RuntimeError(f"{method} {url} → {e.code}\n{detail}")


def git(*args, check=True):
    # core.quotepath=false：中文文件名不能让它转成 "\344\275\277..."，
    # 否则 ls-files 拿到的路径在磁盘上根本找不到（会把已有文件当成删除）。
    return subprocess.run(["git", "-c", "core.quotepath=false"] + list(args),
                          cwd=str(HERE), check=check,
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


def local_commit(ver: str) -> bool:
    """把工作区改动提交到本地 main。返回是否真的产生了新提交。"""
    if not (HERE / ".git").is_dir():
        git("init", "-b", "main")
        print("已初始化 git 仓库")
    git("add", "-A")
    st = git("status", "--porcelain")
    if not st.stdout.strip():
        print("工作区没有变更，跳过提交")
        return False
    git("-c", "user.name=COF Studio", "-c", "user.email=cof@localhost",
        "commit", "-m", f"COF Studio {ver}")
    print(f"已提交 {len(st.stdout.strip().splitlines())} 个变更")
    return True


def push_source_api(repo: str, ver: str) -> None:
    """走 REST 的 Git Data API 提交源码（github.com:443 连不上时用）。

    为什么需要这一条：git push 要连 github.com，很多网络里只有
    api.github.com 可达（实测 github.com 直接超时，api 200）。API 不
    走 git 协议，把文件按 blob 传上去、建一棵树、建一个提交就落库了。

    比对基准不用本地历史，而是**远程当前那棵树**：逐文件比 blob sha
    （`git hash-object` 算出来的和远程一模一样），只传真有差异的，
    本地提交历史跟远程对不上也不影响。这样每次发版都在远程 HEAD 上
    接一个提交，GitHub 那边永远是干净的直线。
    """
    import base64

    ref = api("GET", f"/repos/{repo}/git/ref/heads/main")
    head = ref["object"]["sha"]
    base_tree = api("GET", f"/repos/{repo}/git/commits/{head}")["tree"]["sha"]
    tree = api("GET", f"/repos/{repo}/git/trees/{base_tree}?recursive=1")
    remote = {t["path"]: t["sha"] for t in tree["tree"] if t["type"] == "blob"}
    print(f"远程 HEAD {head[:7]}：{len(remote)} 个文件")

    local_commit(ver)               # 本地也留一条记录（推不上去也不丢）
    # ls-files -s 直接给出暂存区的 mode + blob sha，省得再算一遍
    tracked = {}
    for line in git("ls-files", "-s").stdout.splitlines():
        meta, path = line.split("\t", 1)
        mode, sha, _stage = meta.split()
        tracked[path] = (mode, sha)

    entries = []
    for path, (mode, sha) in tracked.items():
        f = HERE / path
        if not f.is_file():
            continue
        if remote.get(path) == sha:
            continue
        data = f.read_bytes()
        blob = api("POST", f"/repos/{repo}/git/blobs", {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64"})
        entries.append({"path": path, "mode": mode, "type": "blob",
                        "sha": blob["sha"]})
        print(f"  {'~' if path in remote else '+'} {path}  "
              f"{len(data) / 1024:.0f} KB")

    for path in sorted(set(remote) - set(tracked)):
        if (HERE / path).is_file():
            # 本地明明还在却没被跟踪（多半是路径编码没对上）——
            # 宁可少删，也别把远程的文件删掉
            print(f"  ! {path} 本地存在但未被跟踪，跳过删除")
            continue
        entries.append({"path": path, "mode": "100644", "type": "blob",
                        "sha": None})
        print(f"  − {path}")
    if not entries:
        print("远程内容已经是最新，没有要传的文件")
        return

    new_tree = api("POST", f"/repos/{repo}/git/trees",
                   {"base_tree": base_tree, "tree": entries})["sha"]
    new = api("POST", f"/repos/{repo}/git/commits", {
        "message": f"COF Studio {ver}", "tree": new_tree, "parents": [head]})
    api("PATCH", f"/repos/{repo}/git/refs/heads/main",
        {"sha": new["sha"], "force": False})
    print(f"已推送源码到 {repo}：{len(entries)} 个文件，提交 {new['sha'][:7]}")


def push_source(repo: str, ver: str) -> None:
    """初始化（若需要）、提交、推送。Token 只在命令行里出现，不写进 .git/config。"""
    url = f"https://x-access-token:{token()}@github.com/{repo}.git"
    local_commit(ver)
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
        # tag 引用留着的话，新 Release 会复用指向旧提交的 tag；一并删掉
        api("DELETE", f"/repos/{repo}/git/refs/tags/{tag}", ok=(204,))
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
    if "cof_full_" + ver + ".zip" in missing:
        sys.exit("release/ 里缺整包（先跑 make_release.py --full）：\n  "
                 + "\n  ".join(missing))
    if missing:
        # 补丁可以不发（改了 cof_studio.py 的版本必须整包），
        # latest.json 里没 patch 字段客户端就只露「下载整包」按钮
        print("注意：没有补丁包，本版只发整包 ——", missing)
    for name in [n for n in want if n not in missing]:
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
    ap.add_argument("--no-push", action="store_true",
                    help="跳过 git push（网络不通或源码已用别的方式推上去时用）")
    ap.add_argument("--api-push", action="store_true",
                    help="用 REST API 提交源码（github.com:443 连不上、只有 "
                         "api.github.com 可达时用）")
    ap.add_argument("--desc", default="COF Studio：COF 结构建模与势函数优化（本地网页界面）")
    args = ap.parse_args()

    ver = app_version()
    print(f"版本 {ver} → {args.repo}\n")

    if args.create_repo:
        ensure_repo_created(args.repo, args.desc)
    if args.api_push:
        push_source_api(args.repo, ver)
    elif not args.no_push:
        push_source(args.repo, ver)
    if args.no_release:
        return
    rel = make_release(args.repo, ver, args.notes)
    upload_assets(args.repo, rel, ver)
    print("\n客户端「更新源」填：")
    print(f"  https://github.com/{args.repo}/releases/latest/download")


if __name__ == "__main__":
    main()
