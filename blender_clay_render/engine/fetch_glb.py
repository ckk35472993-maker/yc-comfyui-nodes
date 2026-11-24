# fetch_glb.py —— 开源 3D 模型抓取（render_model.py 白膜控制片原料库）
# 用法:
#   python fetch_glb.py bottle --limit 5            # poly.pizza（默认源，零凭证）
#   python fetch_glb.py --url https://poly.pizza/m/FAHsHFXfTf
#   python fetch_glb.py bottle --limit 5 --out <dir>
#   python fetch_glb.py bottle --source sketchfab  # 需 SKETCHFAB_API_TOKEN（env，不打包）
# 默认库目录: C:/Users/admin/VideoAI_Data/modules/blender_assets（用户素材根，非 git 区）
# 产物: <out>/<slug>.glb 或 <slug>__<uid>/ 目录 + <out>/manifest.json（题名/来源/许可台账）
# 仅 stdlib；static.poly.pizza 直链 + Sketchfab Data API v3（search 无鉴权，download 带 Token）
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

# 默认库目录：VIDEOAI_ASSETS_DIR > $VIDEOAI_DATA/modules/blender_assets > ~/VideoAI_Data/modules/blender_assets
# （本机解析结果即 C:/Users/admin/VideoAI_Data/modules/blender_assets，与旧硬编码一致；
#  其他用户机按各自用户目录解析，Electron 可用环境变量覆盖）
DATA_ROOT = os.environ.get("VIDEOAI_DATA", os.path.expanduser("~/VideoAI_Data"))
DEFAULT_ASSETS_DIR = os.environ.get(
    "VIDEOAI_ASSETS_DIR", os.path.join(DATA_ROOT, "modules", "blender_assets"))

BASE = "https://poly.pizza"
UA = {"User-Agent": "Mozilla/5.0 (fetch_glb; whiteref-assets)"}
GLB_RE = re.compile(r"https://static\.poly\.pizza/([0-9a-f-]{36})\.glb")
MODEL_RE = re.compile(r'href="/m/([A-Za-z0-9_-]+)"')


def http_get(url, timeout=20, headers=None):
    # urllib 在本机实测单请求 ~34s（疑似代理解析/IPv6 回退）；curl 同站 ~2s，优先 curl
    hdrs = dict(UA)
    if headers:
        hdrs.update(headers)
    try:
        cmd = ["curl", "-sL", "--max-time", str(timeout), "-A", UA["User-Agent"]]
        for k, v in hdrs.items():
            if k == "User-Agent":
                continue
            cmd += ["-H", f"{k}: {v}"]
        cmd.append(url)
        r = subprocess.run(cmd, capture_output=True, timeout=timeout + 5)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:
        pass
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_download(url, fpath, max_time=180, headers=None, retries=3,
                  min_speed=10240):
    """文件落盘下载：curl -C - 断点续传 + --speed-limit 低速熔断。
    成功→fpath 就位返回字节数；失败→留 .part 供下次续传，返回 None。
    （S3 等大文件通道必备：http_get 内存抓取一旦超时前功尽弃）"""
    hdrs = dict(UA)
    if headers:
        hdrs.update(headers)
    part = fpath + ".part"
    for _ in range(retries):
        cmd = ["curl", "-sL", "--max-time", str(max_time),
               "--speed-limit", str(min_speed), "--speed-time", "30",
               "-C", "-", "-o", part, "-A", UA["User-Agent"]]
        for k, v in hdrs.items():
            if k != "User-Agent":
                cmd += ["-H", f"{k}: {v}"]
        cmd.append(url)
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=max_time + 15)
        except Exception as e:
            print(f"    [curl {type(e).__name__}] {fpath}", file=sys.stderr)
            break
        if r.returncode == 0 and os.path.exists(part) and os.path.getsize(part) > 0:
            os.replace(part, fpath)
            return os.path.getsize(fpath)
        tail = (r.stderr or b"").decode("utf-8", "replace").strip()[-200:]
        print(f"    [curl rc={r.returncode}] {tail}", file=sys.stderr)
        # rc33 = 签名 URL 过期/服务端不认 Range（S3 签名 300s 到期即此）：
        # 不删 .part——外层拿到新签名 URL 后续传仍有效（同对象）
        if r.returncode not in (18, 28, 33, 52, 55, 56):  # 非可续传网络错误→不重试
            break
    return None


def search_ids(term, limit):
    html = http_get(f"{BASE}/search/{urllib.parse.quote(term)}").decode("utf-8", "replace")
    ids = []
    for m in MODEL_RE.finditer(html):
        if m.group(1) not in ids:
            ids.append(m.group(1))
        if len(ids) >= limit:
            break
    return ids


def parse_model_page(mid):
    html = http_get(f"{BASE}/m/{mid}").decode("utf-8", "replace")
    glb = GLB_RE.search(html)
    if not glb:
        return None
    # og:title = "<Title> - Free Model By <Author>"（poly.pizza 模板固定）
    title = ""
    author = ""
    tm = re.search(r'og:title"?\s*content="([^"]+)"', html)
    if tm:
        raw = tm.group(1).strip()
        m = re.match(r"(.+?)\s*-\s*Free Model By\s+(.+)$", raw)
        if m:
            title, author = m.group(1).strip(), m.group(2).strip()
        else:
            title = raw
    lic = ""
    for pat in ("CC0", "Public Domain", "CC-BY-SA", "CC-BY-NC", "CC-BY"):
        if pat in html:
            lic = pat
            break
    return {
        "id": mid,
        "title": title or mid,
        "glb_url": glb.group(0),
        "license": lic or "unknown",
        "author": author,
        "page": f"{BASE}/m/{mid}",
        "source": "polypizza",
    }


def slugify(s):
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip())[:48].strip("_")
    return s or "model"


def update_manifest(out_dir, entries):
    mp = os.path.join(out_dir, "manifest.json")
    old = []
    if os.path.exists(mp):
        try:
            old = json.load(open(mp, encoding="utf-8"))
        except Exception:
            old = []
    seen = {e.get("id") for e in entries}
    merged = [e for e in old if e.get("id") not in seen] + entries
    json.dump(merged, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return mp


def fetch_one(mid, out_dir):
    info = parse_model_page(mid)
    if not info:
        print(f"  [skip] {mid}: no glb link")
        return None
    fname = slugify(info["title"]) + "__" + mid + ".glb"
    fpath = os.path.join(out_dir, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 20:
        print(f"  [keep] {fname}")
        info["file"] = fname
        return info
    data = http_get(info["glb_url"], timeout=60)
    if data[:4] != b"glTF":
        print(f"  [skip] {mid}: bad magic {data[:8]!r}")
        return None
    with open(fpath, "wb") as f:
        f.write(data)
    info["file"] = fname
    print(f"  [ok] {fname} ({len(data)//1024}KB, {info['license']}) {info['title']}")
    return info


# ─── Sketchfab 源（second source，官方 Data API v3）───
# search 端点无鉴权可用（2026-11-19 实测）；/download 必须 Authorization: Token <key>
# （账号级 personal API token，sketchfab.com/settings/password 页取）。凭据只走 env 不打包。
SF_API = "https://api.sketchfab.com/v3"
SF_TOKEN_ENV = "SKETCHFAB_API_TOKEN"
_SF_META = {}  # uid → search result dict，省一次 /models/{uid} 详情请求
_SF_UID_RE = re.compile(r"[0-9a-f]{32}")
_SF_LIC_MAP = (("CC0", "CC0"), ("Public Domain", "Public Domain"),
               ("NonCommercial", "CC-BY-NC"), ("ShareAlike", "CC-BY-SA"),
               ("NoDerivs", "CC-BY-ND"), ("Attribution", "CC-BY"))


def sketchfab_token():
    return os.environ.get(SF_TOKEN_ENV, "").strip()


def sketchfab_search_ids(term, limit):
    """GET /v3/search?downloadable=true（无鉴权）。返回 uid 列表，元数据进 _SF_META"""
    url = (f"{SF_API}/search?type=models&q={urllib.parse.quote(term)}"
           "&downloadable=true&archives_flavours=false")
    data = json.loads(http_get(url, timeout=30).decode("utf-8", "replace"))
    ids = []
    for m in data.get("results", []):
        if not m.get("isDownloadable") or not m.get("uid"):
            continue
        _SF_META[m["uid"]] = m
        if m["uid"] not in ids:
            ids.append(m["uid"])
        if len(ids) >= limit:
            break
    return ids


def _sf_license(m):
    lab = ((m.get("license") or {}).get("label") or "")
    for k, v in _SF_LIC_MAP:
        if k in lab:
            return v
    return lab or "unknown"


def _sf_entry_path(dpath):
    """Sketchfab zip 解出的入口：优先 scene.gltf，否则首个 *.gltf/*.glb（一层深）"""
    if not os.path.isdir(dpath):
        return None
    names = os.listdir(dpath)
    for want in ("scene.gltf",):
        if want in names:
            return os.path.join(dpath, want)
    for ext in (".gltf", ".glb"):
        for n in names:
            if n.lower().endswith(ext):
                return os.path.join(dpath, n)
    return None


def _extract_sf_zip(data, dpath):
    os.makedirs(dpath, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dpath)
    return _sf_entry_path(dpath)


def fetch_one_sketchfab(uid, out_dir, token=None):
    """GET /v3/models/{uid}/download → 优先 glb 直出单文件（与库平铺约定一致，
    ensure_model 顶层扫描可命中）；无 glb 变体时回退 gltf zip 解目录。
    产物 = <out>/<slug>__<uid>.glb 或 <slug>__<uid>/ 目录；
    manifest file 记文件名或 '<dir>/scene.gltf' 相对路径。"""
    token = token or sketchfab_token()
    if not token:
        print(f"  [skip] {uid}: {SF_TOKEN_ENV} 未设置")
        return None
    m = _SF_META.get(uid)
    if not m:
        try:
            m = json.loads(http_get(f"{SF_API}/models/{uid}", timeout=30).decode("utf-8", "replace"))
        except Exception:
            m = {}
    try:
        dl = json.loads(http_get(f"{SF_API}/models/{uid}/download",
                                 headers={"Authorization": f"Token {token}"},
                                 timeout=30)
                        .decode("utf-8", "replace"))
    except Exception as e:
        print(f"  [skip] {uid}: download api {type(e).__name__}")
        return None
    if isinstance(dl, dict) and dl.get("detail"):
        print(f"  [skip] {uid}: api auth — {dl.get('detail')}")
        return None
    title = m.get("name") or uid
    entry_rel = None

    arc_glb = dl.get("glb") or {}
    arc_zip = dl.get("gltf") or {}
    size_mb = lambda arc: int(arc.get("size") or 0) / 1048576
    if arc_glb.get("url"):
        # 单文件 GLB 变体：直接落盘，免解压（断点续传+大小预检+动态时限）
        mb = size_mb(arc_glb)
        if mb > 96:
            print(f"  [skip] {uid}: glb too big {mb:.0f}MB")
            return None
        fname = slugify(title) + "__" + uid + ".glb"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 20:
            print(f"  [keep] {fname}")
        else:
            mt = max(120, int(mb * 25) + 60)  # 按 ≥40KB/s 保底估算 +60s 余量
            got = None
            for _try in range(3):  # 签名 URL 300s 过期：失败换新签名再续传 .part
                got = http_download(arc_glb["url"], fpath, max_time=mt)
                if got:
                    break
                try:
                    dl2 = json.loads(http_get(
                        f"{SF_API}/models/{uid}/download",
                        headers={"Authorization": f"Token {token}"},
                        timeout=30).decode("utf-8", "replace"))
                    arc_glb = dl2.get("glb") or {}
                except Exception:
                    break
                if not arc_glb.get("url"):
                    break
            if not got or open(fpath, "rb").read(4) != b"glTF":
                print(f"  [skip] {uid}: glb download failed/bad magic")
                return None
            print(f"  [ok] {fname} ({got//1024}KB, {_sf_license(m)}) {title}")
        entry_rel = fname
    elif arc_zip.get("url"):
        # 回退：gltf zip 解成 <slug>__<uid>/ 目录（gltf+bin+textures 同居）
        mb = size_mb(arc_zip)
        if mb > 96:
            print(f"  [skip] {uid}: zip too big {mb:.0f}MB")
            return None
        dname = slugify(title) + "__" + uid
        dpath = os.path.join(out_dir, dname)
        entry = _sf_entry_path(dpath)
        if entry and os.path.getsize(entry) > 20:
            print(f"  [keep] {dname}")
            entry_rel = dname + "/" + os.path.basename(entry)
        else:
            zpath = dpath + ".zip"
            mt = max(120, int(mb * 25) + 60)
            got = http_download(arc_zip["url"], zpath, max_time=mt)
            if not got or open(zpath, "rb").read(2) != b"PK":
                print(f"  [skip] {uid}: zip download failed/bad magic")
                return None
            with open(zpath, "rb") as f:
                data = f.read()
            entry = _extract_sf_zip(data, dpath)
            os.remove(zpath)
            if not entry:
                print(f"  [skip] {uid}: no gltf/glb in zip")
                return None
            entry_rel = dname + "/" + os.path.basename(entry)
            print(f"  [ok] {dname}/ ({got//1024}KB, {_sf_license(m)}) {title}")
    else:
        print(f"  [skip] {uid}: no archive url")
        return None
    return {"id": uid, "title": title, "license": _sf_license(m),
            "author": (m.get("user") or {}).get("displayName")
                      or (m.get("user") or {}).get("username", ""),
            "page": m.get("viewerUrl") or f"https://sketchfab.com/3d-models/{uid}",
            "file": entry_rel, "source": "sketchfab"}


def main():
    for _s in (sys.stdout, sys.stderr):  # 管道端强制 UTF-8，防标题生僻字/emoji 炸 print
        if not _s.isatty() and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    out_dir = DEFAULT_ASSETS_DIR
    limit = 5
    url_mode = None
    source = "polypizza"
    term = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--out":
            out_dir = args[i + 1]; i += 2
        elif a == "--limit":
            limit = int(args[i + 1]); i += 2
        elif a == "--url":
            url_mode = args[i + 1]; i += 2
        elif a == "--source":
            source = args[i + 1].strip().lower(); i += 2
        elif a.startswith("-"):
            i += 1
        else:
            term = a; i += 1
    if source not in ("polypizza", "sketchfab"):
        print(f"FATAL bad --source: {source} (valid: polypizza|sketchfab)")
        return 2
    if source == "sketchfab" and not sketchfab_token():
        print(f"sketchfab source needs {SF_TOKEN_ENV} env（账号 API Token，不打包）")
        return 2
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir) \
        if not os.path.isabs(out_dir) else out_dir
    os.makedirs(out_dir, exist_ok=True)

    if url_mode:
        if source == "sketchfab":
            m = _SF_UID_RE.search(url_mode)   # …-slug-<32hex> 或 /models/<uid>
            ids = [m.group(0)] if m else []
        else:
            m = re.search(r"/m/([A-Za-z0-9_-]+)", url_mode)
            ids = [m.group(1)] if m else []
    elif term:
        ids = (sketchfab_search_ids if source == "sketchfab"
               else search_ids)(term, limit)
    else:
        print(__doc__)
        return 2
    if not ids:
        print("no model ids resolved")
        return 1
    print(f"ids: {ids}")

    fetch_fn = fetch_one_sketchfab if source == "sketchfab" else fetch_one
    got = []
    for mid in ids:
        try:
            info = fetch_fn(mid, out_dir)
            if info:
                got.append(info)
        except Exception as e:
            print(f"  [err] {mid}: {e}")
        time.sleep(0.4)  # 礼貌限速
    if got:
        mp = update_manifest(out_dir, got)
        print(f"manifest: {mp} ({len(got)} new)")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
