# ensure_model.py —— 白膜模型「有则用、无则下」封装（fetch-on-miss）
# 用法:
#   python ensure_model.py <查询词或模型路径> [--json]
#   python ensure_model.py yacht --spec '{"motion":"orbit360","frames":72}' --blender <blender.exe>
# 流程:
#   1. 参数已是存在的文件路径 → 直接用
#   2. 本地库关键词匹配（manifest 题名 + 文件名）→ 命中返回
#   3. 未命中 → fetch_glb 搜索 poly.pizza + 下载 → 按标题匹配度选最优
#   4. --spec 提供时顺带调 Blender 渲白膜（--blender 或环境变量 BLENDER_EXE）
# 输出契约: 末行 MODEL {"path","title","license","source":local|fetched|file}
# 依赖: 纯 stdlib + curl（Win10+ 自带）+ fetch_glb.py（复用其全部下载/解析函数）
import contextlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_glb  # 复用：search_ids/fetch_one/manifest/http_get/DEFAULT_ASSETS_DIR
import blender_resolve  # 复用 get_bin_path 哲学的探测链：显式>env>打包>自动>""

ASSETS_DIR = fetch_glb.DEFAULT_ASSETS_DIR
MODEL_EXTS = (".glb", ".gltf", ".fbx", ".obj", ".stl", ".ply")
FETCH_DEADLINE_S = 240  # 联网抓取总预算：超时仍返回已抓到的最优，防 Electron 后台任务悬挂


def tokenize(s):
    # 拉丁字母/数字 ≥2 字符成词；中日韩字符单个即成词（中文文件名/查询可命中）
    out = []
    for t in re.split(r"[^a-z0-9一-鿿]+", s.lower()):
        if not t:
            continue
        if len(t) >= 2 or any('一' <= c <= '鿿' for c in t):
            out.append(t)
    return out


def _tok_match(q, tok):
    """词级匹配：整词相等或简单复数差一位（cat↔cats）。子串不算——cat 不许命中 castle"""
    return q == tok or q == tok + "s" or q == tok + "es" or tok == q + "s" or tok == q + "es"


def _score(q_tokens, hay):
    hay_tokens = tokenize(hay)
    # 正向复合：query 单词 = hay 连续 2~3 词拼接（"lawnmower" ↔ "lawn mower"）
    spans = hay_tokens + ["".join(hay_tokens[i:i + n]) for n in (2, 3)
                          for i in range(len(hay_tokens) - n + 1)]
    score = sum(1 for q in q_tokens if any(_tok_match(q, t) for t in spans))
    # 反向复合：query 多词拼接 = hay 单词（"sports car" ↔ "sportscar"）
    if len(q_tokens) > 1 and score < len(q_tokens):
        qj = "".join(q_tokens)
        if any(_tok_match(qj, t) for t in hay_tokens):
            score = len(q_tokens)
    return score


def local_candidates(query, assets_dir):
    """本地库按词匹配打分，返回 [(score, path, title)] 降序"""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    hits = []
    manifest = {}
    mp = os.path.join(assets_dir, "manifest.json")
    if os.path.exists(mp):
        try:
            for e in json.load(open(mp, encoding="utf-8")):
                manifest[e.get("file", "")] = e
        except Exception:
            pass
    if not os.path.isdir(assets_dir):
        return []
    for fn in os.listdir(assets_dir):
        if os.path.splitext(fn)[1].lower() not in MODEL_EXTS or fn.startswith("._"):
            continue
        title = (manifest.get(fn) or {}).get("title") or ""
        hay = title + " " + os.path.splitext(fn)[0]
        score = _score(q_tokens, hay)
        if score:
            # 同分下标题全等者优先："cat" 该命中 "Cat" 而非 "Bowls Cat"
            exact = int(title.strip().lower() == query.strip().lower())
            hits.append((score, exact, os.path.join(assets_dir, fn), title or fn))
    # 目录型资产（Sketchfab zip 解出 file="Dir__uid/scene.gltf"）：平铺扫描扫不到，走 manifest
    seen = {h[2] for h in hits}
    for e in manifest.values():
        rel = (e or {}).get("file") or ""
        if not rel or ("/" not in rel and "\\" not in rel):
            continue
        fp = os.path.join(assets_dir, rel)
        if fp in seen or not os.path.isfile(fp):
            continue
        title = e.get("title") or ""
        hay = title + " " + os.path.basename(os.path.dirname(fp))
        score = _score(q_tokens, hay)
        if score:
            exact = int(title.strip().lower() == query.strip().lower())
            hits.append((score, exact, fp, title or os.path.dirname(fp)))
    hits.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return hits


def ensure(query, assets_dir=ASSETS_DIR, fetch_limit=8, fetch_deadline_s=FETCH_DEADLINE_S):
    """返回 (model_dict, source)。未命中时联网下载并按标题匹配择优"""
    if os.path.isfile(query):
        return {"path": os.path.abspath(query), "title": os.path.basename(query),
                "license": "", "source": "file"}, "file"
    # 形如模型文件名/路径但不存在 → 先按文件名查库（传 "Cat__xxx.glb" 应命中库里同名文件），
    # 库里也没有才算 file_missing，不当搜索词送外网
    if os.path.splitext(query)[1].lower() in MODEL_EXTS:
        lib_hit = os.path.join(assets_dir, os.path.basename(query))
        if os.path.isfile(lib_hit):
            ent = _manifest_entry(assets_dir, lib_hit) or {}
            return {"path": lib_hit, "title": ent.get("title") or os.path.basename(query),
                    "license": ent.get("license", ""), "source": "local"}, "local"
        return None, "file_missing"

    hits = local_candidates(query, assets_dir)
    q_tokens = tokenize(query)
    if not q_tokens:
        return None, "empty_query"
    # 全词命中直接用；部分命中但最优分>=2 词或独占词也用
    if hits and (hits[0][0] == len(q_tokens) or hits[0][0] >= 2):
        ent = _manifest_entry(assets_dir, hits[0][2])
        return {"path": hits[0][2], "title": hits[0][3],
                "license": (ent or {}).get("license", ""),
                "source": "local"}, "local"

    os.makedirs(assets_dir, exist_ok=True)
    deadline = time.monotonic() + fetch_deadline_s
    best = None
    fetched_all = []
    try:
        ids = fetch_glb.search_ids(query, fetch_limit)
    except Exception as e:
        # 主源网络死不该连累备源——按 0 结果继续走 sketchfab 兜底
        print(f"  [warn] poly.pizza search failed: {type(e).__name__}",
              file=sys.stderr)
        ids = []
    if ids:
        best, fetched_all = _fetch_best(ids, fetch_glb.fetch_one, assets_dir,
                                      q_tokens, deadline, query, fetch_deadline_s)
    # second source：poly.pizza 无合格结果且配了 token → Sketchfab（opportunistic，
    # 其失败不升级——主源已 miss，bonus 源挂了照样按 miss 报）
    if (not best or best[0][0] < 1) and fetch_glb.sketchfab_token():
        try:
            ids2 = fetch_glb.sketchfab_search_ids(query, fetch_limit)
        except Exception:
            ids2 = []
        if ids2:
            tok = fetch_glb.sketchfab_token()
            # 保底预算：poly 段吃光总 deadline 时仍给兜底留 ≥60s，防慢网日二源饿死
            deadline2 = max(deadline, time.monotonic() + 60.0)
            b2, f2 = _fetch_best(
                ids2, lambda mid, d: fetch_glb.fetch_one_sketchfab(mid, d, tok),
                assets_dir, q_tokens, deadline2, query, fetch_deadline_s)
            fetched_all += f2
            if b2 and (not best or b2[0] > best[0]):
                best = b2
    if (not best or best[0][0] < 1) and not fetch_glb.sketchfab_token():
        print(f"  [hint] poly.pizza 无合格结果；设 {fetch_glb.SF_TOKEN_ENV} "
              "可启用 Sketchfab 第二源", file=sys.stderr)
    if fetched_all:
        fetch_glb.update_manifest(assets_dir, fetched_all)
    # overlap<1 = 搜索页侧栏噪音/无结果页兜底链接，宁可 miss 不静默发错模型
    if not best or best[0][0] < 1:
        return None, "miss"
    info = best[1]
    out = {"path": os.path.join(assets_dir, info["file"]),
           "title": info["title"], "license": info["license"],
           "id": info["id"], "source": "fetched"}
    if info.get("source"):
        out["via"] = info["source"]  # polypizza|sketchfab 溯源
    return out, "fetched"


def _fetch_best(ids, fetch_fn, assets_dir, q_tokens, deadline, query, budget_s):
    """按标题匹配度择优下载。返回 (best|(score,info)|None, fetched_all)"""
    best = None
    fetched_all = []  # 全部落盘文件都要进 manifest——未选中的孤儿也要有许可台账
    for mid in ids:
        if time.monotonic() > deadline:
            print(f"  [deadline] fetch budget {budget_s}s exceeded, keep best so far",
                  file=sys.stderr)
            break
        try:
            info = fetch_fn(mid, assets_dir)
        except Exception:
            info = None
        if not info:
            continue
        fetched_all.append(info)
        time.sleep(0.3)  # 与 fetch_glb 主循环同规的礼貌限速
        title_tokens = tokenize(info.get("title", ""))
        overlap = _score(q_tokens, info.get("title", ""))
        exact = int(info.get("title", "").strip().lower() == query.strip().lower())
        score = (overlap, exact, -(len(title_tokens) - overlap))  # 命中多>全等>冗余少
        if best is None or score > best[0]:
            best = (score, info)
        if overlap == len(q_tokens) and len(q_tokens) > 0:
            break
    return best, fetched_all


def _manifest_entry(assets_dir, path):
    mp = os.path.join(assets_dir, "manifest.json")
    # 统一按相对路径比：平铺文件 rel==basename；目录型资产 "Dir__uid/scene.gltf" 也命中
    rel = os.path.relpath(path, assets_dir).replace("\\", "/")
    try:
        for e in json.load(open(mp, encoding="utf-8")):
            if (e.get("file") or "").replace("\\", "/") == rel:
                return e
    except Exception:
        return None
    return None


def render_whiteref(model_path, spec, blender_exe, script_dir, timeout_s=1800):
    rm = os.path.join(script_dir, "render_model.py")
    spec = dict(spec)
    spec["model_path"] = model_path
    cmd = [blender_exe, "-b", "--python", rm, "--", "--spec", json.dumps(spec)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "render_timeout", "timeout_s": timeout_s}
    m = re.search(r'SELFCHECK (\{.*\})', r.stdout or "")
    out = {"returncode": r.returncode,
           "selfcheck": json.loads(m.group(1)) if m else None}
    if r.returncode != 0:  # 失败时带回 stderr 尾巴供排障，Electron 上层不用翻日志
        out["stderr_tail"] = (r.stderr or "")[-1500:]
    return out


def main():
    # 管道端按系统 locale(GBK) 编码——模型标题带 emoji/生僻字会 UnicodeEncodeError
    # 崩在最后一行 print，连 MODEL 契约行都打不出。非 tty 强制 UTF-8。
    for _s in (sys.stdout, sys.stderr):
        if not _s.isatty() and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    out_json = "--json" in args  # 激活：进度输出挪 stderr，stdout 只留末行 JSON 契约
    spec = None
    blender = os.environ.get("BLENDER_EXE", "")
    fetch_deadline_s = FETCH_DEADLINE_S
    query = None
    i = 0
    try:
        while i < len(args):
            a = args[i]
            if a == "--spec":
                spec = json.loads(args[i + 1]); i += 2
            elif a == "--blender":
                blender = args[i + 1]; i += 2
            elif a == "--out":
                global ASSETS_DIR; ASSETS_DIR = os.path.abspath(args[i + 1]); i += 2
            elif a == "--fetch-deadline":
                fetch_deadline_s = int(args[i + 1]); i += 2
            elif a.startswith("-"):
                i += 1
            else:
                query = a; i += 1
    except Exception as e:  # 缺值/坏 JSON 也走结构化契约，不裸 traceback
        print("MODEL " + json.dumps(
            {"ok": False, "reason": f"bad_args:{e.__class__.__name__}"},
            ensure_ascii=False))
        return 2
    if not query:
        print(__doc__)
        return 2

    try:
        if out_json:
            with contextlib.redirect_stdout(sys.stderr):
                model, source = ensure(query, ASSETS_DIR,
                                       fetch_deadline_s=fetch_deadline_s)
        else:
            model, source = ensure(query, ASSETS_DIR,
                                   fetch_deadline_s=fetch_deadline_s)
    except Exception as e:
        model, source = None, f"error:{type(e).__name__}"
        print(f"  [err] ensure crashed: {e}", file=sys.stderr)
    if not model:
        reason = source if isinstance(source, str) and (source.startswith("error")
                          or source in ("file_missing", "empty_query")) \
                 else "not_found_anywhere"
        print("MODEL " + json.dumps({"ok": False, "query": query, "reason": reason},
                                    ensure_ascii=False))
        return 1
    result = {"ok": True, "model": model}
    if spec is not None:  # 空 dict {} 也是合法 spec（吃 render_model 默认值），不能按 falsy 跳
        blender = blender_resolve.resolve_blender(blender)  # --blender 优先，env/打包/自动探测兜底
        if not blender:
            result["render"] = {"ok": False, "reason": "blender_exe_missing",
                                "hint": "传 --blender <exe> 或设 BLENDER_EXE"}
        else:
            rr = render_whiteref(model["path"], spec, blender,
                                 os.path.dirname(os.path.abspath(__file__)))
            if "returncode" in rr:
                rr["ok"] = rr["returncode"] == 0 and \
                    bool((rr["selfcheck"] or {}).get("ok"))
            result["render"] = rr
    print("MODEL " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
