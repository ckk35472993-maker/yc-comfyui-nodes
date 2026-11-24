# blender_resolve.py —— 用户机 Blender 可执行文件解析（Win/Mac）
# 优先级：显式 > env(VIDEOAI_BLENDER_PATH/BLENDER_EXE) > 打包便携 > 系统探测 > ""
# 每个候选先过 `blender --version` 实测（15s），结果进程内缓存。
# 契约：resolve_blender() 仍返回 path 字符串（存量调用零破坏）；
#       resolve_blender_info() 返回 {ok,exe,version,source,platform} 给设置页/Bridge。
import glob
import os
import platform
import re
import shutil
import subprocess

_ENV_KEYS = ("VIDEOAI_BLENDER_PATH", "BLENDER_EXE")
# path -> (ok: bool, version: str)
_VERSION_CACHE = {}

_PIN_MAJOR_HINT = "5"  # 产品钉 5.x LTS（use_nodes 将在 6.0 移除）；仅提示，不硬拒


def _plat():
    return platform.system()


def _data_root():
    return os.environ.get(
        "VIDEOAI_DATA",
        os.path.expanduser(os.path.join("~", "VideoAI_Data")),
    )


def normalize_blender_path(path):
    """把 .app 包 / 目录 / 引号路径归一成可执行文件路径（不存在也返回归一后的猜测）。"""
    if not path:
        return ""
    p = os.path.expanduser(str(path).strip().strip('"').strip("'"))
    if not p:
        return ""
    p = os.path.normpath(p)
    # Mac: 用户选了 Blender.app
    if p.endswith(".app") or p.lower().endswith(".app"):
        inner = os.path.join(p, "Contents", "MacOS", "Blender")
        return inner
    # Mac: 目录里是 .app
    app = os.path.join(p, "Blender.app", "Contents", "MacOS", "Blender")
    if os.path.isdir(os.path.join(p, "Blender.app")):
        return app
    if os.path.isdir(p):
        if _plat() == "Windows":
            exe = os.path.join(p, "blender.exe")
            if os.path.isfile(exe):
                return exe
        else:
            for rel in (
                os.path.join("Blender.app", "Contents", "MacOS", "Blender"),
                "Blender",
                "blender",
            ):
                cand = os.path.join(p, rel)
                if os.path.isfile(cand):
                    return cand
    return p


def _bundled_candidates():
    """热更/便携落点：Win=blender.exe；Mac=Blender.app/Contents/MacOS/Blender。"""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.join(_data_root(), "modules", "blender-portable"),
        os.path.join(_data_root(), "modules", "blender"),
        os.path.join(here, "blender-portable"),
        os.path.join(here, "blender"),
    ]
    cands = []
    for root in roots:
        if _plat() == "Darwin":
            cands += [
                os.path.join(root, "Blender.app", "Contents", "MacOS", "Blender"),
                os.path.join(root, "Contents", "MacOS", "Blender"),
                os.path.join(root, "Blender"),
                os.path.join(root, "blender"),
            ]
        elif _plat() == "Windows":
            cands += [
                os.path.join(root, "blender.exe"),
                os.path.join(root, "blender"),
            ]
        else:
            cands += [
                os.path.join(root, "blender"),
                os.path.join(root, "blender.exe"),
            ]
    return cands


def _registry_candidates():
    """Windows 注册表：Uninstall InstallLocation/DisplayIcon + HKCR blendfile 关联"""
    cands = []
    try:
        import winreg
    except ImportError:
        return cands
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            root = winreg.OpenKey(
                hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
        except OSError:
            continue
        for i in range(winreg.QueryInfoKey(root)[0]):
            try:
                k = winreg.OpenKey(root, winreg.EnumKey(root, i))
                name = (winreg.QueryValueEx(k, "DisplayName")[0] or "")
                if "blender" not in name.lower():
                    continue
                for vname in ("InstallLocation", "DisplayIcon"):
                    try:
                        p = (winreg.QueryValueEx(k, vname)[0] or "").strip().strip('"')
                    except OSError:
                        continue
                    if p.lower().endswith("blender.exe"):
                        cands.append(p)
                    elif p and os.path.isdir(p):
                        cands.append(os.path.join(p, "blender.exe"))
                    elif p and os.path.isdir(os.path.dirname(p)):
                        cands.append(os.path.join(os.path.dirname(p), "blender.exe"))
            except OSError:
                continue
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                            r"blendfile\shell\open\command") as k:
            cmd = (winreg.QueryValueEx(k, "")[0] or "")
            m = cmd.strip().strip('"').split('"')[0]
            if m.lower().endswith("blender.exe"):
                cands.append(m)
    except OSError:
        pass
    return cands


def _auto_candidates():
    cands = []
    w = shutil.which("blender")
    if w:
        cands.append(w)
    if _plat() == "Windows":
        roots = {os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")}
        for root in roots:
            if not root:
                continue
            cands += glob.glob(os.path.join(
                root, "Blender Foundation", "*", "blender.exe"))
            cands += glob.glob(os.path.join(root, "Blender*", "blender.exe"))
            cands += glob.glob(os.path.join(
                root, "Steam", "steamapps", "common", "Blender", "blender.exe"))
        cands += _registry_candidates()
    elif _plat() == "Darwin":
        cands += glob.glob("/Applications/Blender.app/Contents/MacOS/Blender")
        cands += glob.glob("/Applications/Blender*/Contents/MacOS/Blender")
        for p in (
            "/opt/homebrew/bin/blender",
            "/usr/local/bin/blender",
            os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
            os.path.expanduser("~/blender/blender"),
        ):
            if os.path.isfile(p):
                cands.append(p)
    else:
        for p in ("/usr/bin/blender", "/usr/local/bin/blender",
                  "/snap/bin/blender", os.path.expanduser("~/blender/blender")):
            if os.path.isfile(p):
                cands.append(p)
    return cands


def _parse_version(stdout_bytes):
    text = (stdout_bytes or b"").decode("utf-8", "replace")
    m = re.search(r"Blender\s+(\d+\.\d+(?:\.\d+)?)", text)
    return m.group(1) if m else ""


def probe_blender(path):
    """返回 (ok, version)。缓存命中不重复起进程。"""
    path = normalize_blender_path(path)
    if not path or not os.path.isfile(path):
        return False, ""
    if path in _VERSION_CACHE:
        return _VERSION_CACHE[path]
    try:
        r = subprocess.run([path, "--version"], capture_output=True, timeout=15)
        ver = _parse_version(r.stdout)
        ok = r.returncode == 0 and (b"Blender" in (r.stdout or b"") or bool(ver))
    except Exception:
        ok, ver = False, ""
    _VERSION_CACHE[path] = (ok, ver)
    return ok, ver


def is_runnable_blender(path):
    """实跑 `blender --version` 验收（15s 超时），结果缓存——同路径不重复起进程"""
    ok, _ = probe_blender(path)
    return ok


def _iter_candidates(explicit=""):
    """yield (path, source) source ∈ explicit|env|portable|auto"""
    if explicit:
        yield normalize_blender_path(explicit), "explicit"
    for k in _ENV_KEYS:
        v = (os.environ.get(k) or "").strip()
        if v:
            yield normalize_blender_path(v), "env"
    for c in _bundled_candidates():
        yield normalize_blender_path(c), "portable"
    for c in _auto_candidates():
        yield normalize_blender_path(c), "auto"


def resolve_blender(explicit=""):
    """返回可用 blender 可执行路径；全落空返回 ""。存量契约不变。"""
    info = resolve_blender_info(explicit)
    return info.get("exe") or ""


def resolve_blender_info(explicit=""):
    """
    设置页/Bridge 用：
      {ok, exe, version, source, platform, pin_hint}
    source: explicit|env|portable|auto|""
    """
    seen = set()
    for c, source in _iter_candidates(explicit):
        if not c:
            continue
        key = os.path.normpath(c)
        # Win 大小写不敏感；Mac 保留原路径作 key 避免误并
        ck = key.lower() if _plat() == "Windows" else key
        if ck in seen:
            continue
        seen.add(ck)
        ok, ver = probe_blender(c)
        if ok:
            return {
                "ok": True,
                "exe": c,
                "version": ver,
                "source": source,
                "platform": _plat(),
                "pin_hint": _PIN_MAJOR_HINT,
            }
    return {
        "ok": False,
        "exe": "",
        "version": "",
        "source": "",
        "platform": _plat(),
        "pin_hint": _PIN_MAJOR_HINT,
    }


def build_blender_python_cmd(exe, script_path, script_args=None):
    """
    用 Blender 自带 CPython 跑 host/脚本（无系统 python）：
      blender -b --python <script> -- <script_args...>
    Electron/Bridge 只应 spawn 这一条，禁止渲染进程自己找 python。
    """
    exe = normalize_blender_path(exe)
    if not exe or not script_path:
        return []
    cmd = [exe, "-b", "--python", os.path.abspath(script_path), "--"]
    if script_args:
        cmd.extend(list(script_args))
    return cmd


if __name__ == "__main__":
    import json
    import sys
    explicit = ""
    as_json = False
    for a in sys.argv[1:]:
        if a == "--json":
            as_json = True
        elif not a.startswith("-"):
            explicit = a
    info = resolve_blender_info(explicit)
    if as_json:
        print("BLENDER " + json.dumps(info, ensure_ascii=False))
    else:
        print("BLENDER " + (info["exe"] if info.get("ok") else "<none>"))
        if info.get("version"):
            print("VERSION " + info["version"])
        if info.get("source"):
            print("SOURCE " + info["source"])
    sys.exit(0 if info.get("ok") else 1)
