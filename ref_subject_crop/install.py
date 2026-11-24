import importlib
import sys

# cv2 是 ComfyUI 生态常见依赖（controlnet_aux 等自带）；缺失时节点照样注册，
# 运行时走 fail-open 透传。这里只做提示不强制 pip，避免在受限环境(RunningHub)安装失败拖垮加载。
for pkg, mod in [("opencv-python", "cv2"), ("numpy", "numpy"), ("torch", "torch")]:
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"[ref_subject_crop] warn: {pkg} 缺失，节点将以透传降级运行", file=sys.stderr)
