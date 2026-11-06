#!/usr/bin/env python3
"""make_web.py —— 从 ui.html 生成 COF Studio 网页版单文件。

网页版架构（界面与计算分离）：
    浏览器 ──打开──> 网页界面（本文件，可托管在任何静态服务器/GitHub Pages，
                      甚至直接双击本地打开）
       │
       └──fetch──> http://127.0.0.1:8770  访问者本机的 COF Studio 引擎
                   （建模/XRD/优化全在本机跑，用自己的 CPU/GPU）

生成逻辑：
  1. 在主脚本前注入"引擎门控"：并发探测 127.0.0.1:8770-8789，
     找到引擎 → 设 window.__API_BASE__（ui.html 里的 fetch 包装会自动接上），
     并按 /api/version 的 capabilities 做特性检测；
     找不到 → 整页换成"下载本机引擎"引导。
  2. 把 ui.html 的主脚本包进 async IIFE，等引擎门控结束再启动。

单一代码源原则：界面永远从 ui.html 生成，改界面后重跑本脚本即可，
不要手改生成出来的 cof_studio_web.html。

用法：
    python make_web.py            # 生成 web/cof_studio_web.html
    python make_web.py --open     # 生成后用默认浏览器打开
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "ui.html"
OUT_DIR = ROOT / "web"
OUT = OUT_DIR / "cof_studio_web.html"

# 引擎下载引导（找不到本机引擎时展示）
DL_PAGE = "https://github.com/1783722112/COF-Studio/releases/latest"

# 主脚本起始锚点（ui.html 中唯一的 <script>，make_web 依赖这个标记）
ANCHOR_OPEN = "<script>"
ANCHOR_CLOSE = "</script>"

# --------------------------------------------------------------------------
# 注入的引擎门控脚本（GATE_TEMPLATE）+ 主脚本包装尾部（TAIL_JS）。
# 注意：这是一段 JS 源码文本，括号必须配对。
# --------------------------------------------------------------------------
TAIL_JS = "\n})();\n"


def build() -> str:
    src = SRC.read_text(encoding="utf-8")

    if src.count(ANCHOR_OPEN) != 1 or src.count(ANCHOR_CLOSE) != 1:
        raise SystemExit(
            f"ui.html 结构变了：期望恰好 1 对 <script>…</script>，"
            f"现在开 {src.count(ANCHOR_OPEN)} 个 / 关 {src.count(ANCHOR_CLOSE)} 个。"
            "请同步修改 make_web.py 的锚点。")

    gate = GATE_TEMPLATE.replace("__DL_PAGE__", DL_PAGE)
    # 顺序很重要：先给主脚本补 IIFE 收尾，再注入头部。
    # 反过来的话，"第一个 </script>" 会匹配到门控自己的关闭标签，拼接就错了。
    out = src.replace(ANCHOR_CLOSE, TAIL_JS + ANCHOR_CLOSE, 1)
    out = out.replace(ANCHOR_OPEN, gate, 1)
    return out


GATE_TEMPLATE = r"""<script>
/* ==========================================================================
   COF Studio 网页版 · 本机引擎门控（由 make_web.py 注入，勿手改）
   ========================================================================== */
window.__ENGINE_READY__ = (async () => {
  const zh = ((navigator.language || "zh").toLowerCase().startsWith("zh"));
  const T = zh ? {
    probing: "正在连接本机计算引擎…",
    notFound: "未检测到本机计算引擎",
    notFoundHint: "网页界面只负责显示，建模、XRD 模拟和势函数优化都在你自己电脑上的引擎里运行（数据不出本机）。请先下载并启动 COF Studio 桌面版，然后回到本页。",
    download: "下载本机引擎（Windows）",
    retry: "重新检测",
    connected: "本机引擎已连接",
  } : {
    probing: "Connecting to local compute engine…",
    notFound: "Local compute engine not found",
    notFoundHint: "This page is only the interface — structure building, XRD simulation and potential optimization all run in the engine on your own machine (data never leaves it). Download and start COF Studio Desktop, then come back to this page.",
    download: "Download local engine (Windows)",
    retry: "Detect again",
    connected: "Local engine connected",
  };

  // ---------- 探测遮罩 ----------
  const ov = document.createElement("div");
  ov.id = "webGate";
  ov.style.cssText = [
    "position:fixed", "inset:0", "z-index:99999",
    "display:flex", "align-items:center", "justify-content:center",
    "background:#f5f6f8", "font-family:'Segoe UI','Microsoft YaHei',sans-serif",
  ].join(";");
  ov.innerHTML = '<div style="text-align:center;max-width:520px;padding:32px">' +
    '<div style="font-size:28px;font-weight:700;color:#1c2333;letter-spacing:.5px">COF Studio <span style="color:#5b8def">Web</span></div>' +
    '<div id="webGateBody" style="margin-top:22px;color:#5b6474;font-size:14px;line-height:1.8"></div></div>';
  document.body.appendChild(ov);
  const gateBody = ov.querySelector("#webGateBody");
  const say = (html) => { gateBody.innerHTML = html; };
  say('<div style="display:inline-block;width:22px;height:22px;border:3px solid #d6dae2;border-top-color:#5b8def;border-radius:50%;animation:cfsSpin 0.9s linear infinite;vertical-align:-6px;margin-right:10px"></div>' + T.probing);
  const gst = document.createElement("style");
  gst.textContent = "@keyframes cfsSpin{to{transform:rotate(360deg)}}";
  document.head.appendChild(gst);

  // ---------- 并发探测 8770-8789（引擎端口被占会自动后移） ----------
  const probe = async (base) => {
    const ctl = new AbortController();
    const tm = setTimeout(() => ctl.abort(), 2500);
    try {
      const r = await fetch(base + "/api/version", {
        signal: ctl.signal, cache: "no-store" });
      if (!r.ok) throw new Error("http " + r.status);
      const info = await r.json();
      if (!info || !info.version) throw new Error("bad payload");
      return { base: base, info: info };
    } finally { clearTimeout(tm); }
  };
  let eng = null;
  try {
    const bases = [];
    for (let p = 8770; p <= 8789; p++) bases.push("http://127.0.0.1:" + p);
    eng = await Promise.any(bases.map(probe));
  } catch (e) { eng = null; }

  if (!eng) {
    // ---------- 未装引擎：引导页，主脚本不会再启动 ----------
    say('<div style="font-size:18px;font-weight:600;color:#b3392e;margin-bottom:12px">' + T.notFound + '</div>' +
        '<div style="margin-bottom:20px">' + T.notFoundHint + '</div>' +
        '<div style="display:flex;gap:12px;justify-content:center">' +
        '<a href="__DL_PAGE__" target="_blank" rel="noopener" style="background:#3b6fe0;color:#fff;text-decoration:none;padding:10px 22px;border-radius:8px;font-weight:600">' + T.download + '</a>' +
        '<button id="webGateRetry" style="background:#fff;border:1px solid #c8cdd8;color:#333;padding:10px 22px;border-radius:8px;cursor:pointer;font-weight:600">' + T.retry + '</button></div>');
    ov.querySelector("#webGateRetry").onclick = () => location.reload();
    throw new Error("COF Studio Web: local engine not found");
  }

  // ---------- 找到引擎：挂基址 + 能力清单，收掉遮罩 ----------
  window.__API_BASE__ = eng.base;
  window.__ENGINE_INFO__ = eng.info;
  const caps = Array.isArray(eng.info.capabilities) ? eng.info.capabilities : [];
  // 特性检测：引擎没有的能力，对应 data-cap 元素直接隐藏（"API 只加不改"的界面侧配套）
  for (const el of document.querySelectorAll("[data-cap]")) {
    if (!caps.includes(el.dataset.cap)) el.style.display = "none";
  }
  ov.remove();

  // 常驻小徽标：让用户知道现在算力来自本机引擎
  const chip = document.createElement("div");
  chip.id = "webEngineChip";
  chip.title = "COF Studio Web · compute via local engine";
  chip.style.cssText = [
    "position:fixed", "left:12px", "bottom:12px", "z-index:99998",
    "background:#eef4ff", "border:1px solid #c4d5f7", "color:#2b4d9b",
    "font-size:12px", "padding:5px 12px", "border-radius:999px",
    "font-family:'Segoe UI','Microsoft YaHei',sans-serif",
    "box-shadow:0 2px 8px rgba(30,60,140,.10)", "user-select:none",
  ].join(";");
  chip.textContent = "⚡ " + T.connected + " · v" + eng.info.version + " · " + eng.base.replace("http://", "");
  document.body.appendChild(chip);
})();
</script>
<script>
/* ==== 主脚本（来自 ui.html）：等引擎门控出结果再启动 ==== */
(async () => {
"use strict";
await window.__ENGINE_READY__;
"""

ANCHOR_OPEN_WRAP = ""  # GATE_TEMPLATE 已自带 <script>(async…，主脚本原样接在后面


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 COF Studio 网页版单文件")
    ap.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    ap.add_argument("--out", default=str(OUT), help="输出路径")
    args = ap.parse_args()

    html = build()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    # 自检：门控里 Promise.any / fetch / 引擎探测关键字都得在
    for token in ("__ENGINE_READY__", "/api/version", "127.0.0.1:",
                  "(async () => {"):
        if token not in html:
            raise SystemExit(f"自检失败：生成文件里找不到 {token!r}")
    kb = len(html.encode("utf-8")) // 1024
    print(f"[网页版] 已生成 {out_path}（{kb} KB）")
    if args.open:
        import webbrowser
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
