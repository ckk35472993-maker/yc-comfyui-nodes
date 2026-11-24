"""VLObjects_SaveText：元素分层工作流的标签落盘节点。

为什么需要它：pysssss SaveText / WAS_Text_Save 写盘后只回 ui.text/ui.string，
不进 /history outputs.images → comfyui_proxy_daemon._collect_local_images 收集不到，
labels.txt 到不了 result。

机制复刻 ComfyUI-CapRL save_txt（nodes.py:269-278）：节点把写出的文件以
{"ui": {"images": [{filename, subfolder, type:"output"}]}} 声明，history 记录后
daemon 收集时命中 TEXT_RESULT_EXTS('.txt'/.json/.md/...) → text_content 内联进
comfyui_completed 消息 + 后台上传对象存储，前端即时拿到名词串正文。
"""

import os

import folder_paths


class VLObjects_SaveText:
    """STRING → output/{subfolder}/{prefix}_{seq:05d}_.{ext}，并在 ui.images 声明产物。"""

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save"
    CATEGORY = "AILab/vl_objects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
                "filename_prefix": ("STRING", {"default": "vl_objects/labels"}),
                # 扩展名可选：manifest JSON 走 .json（同在 TEXT_RESULT_EXTS 收集名单内）
                "extension": (["txt", "json", "md"], {"default": "txt"}),
            },
        }

    def save(self, text, filename_prefix, extension="txt"):
        ext = extension if extension in ("txt", "json", "md") else "txt"
        # 与 SaveImage 同款的 {prefix}_{seq:05d}_ 递增命名，防并发/连跑覆盖
        subfolder = os.path.dirname(filename_prefix)
        prefix = os.path.basename(filename_prefix)
        out_dir = folder_paths.get_output_directory()
        abs_dir = os.path.join(out_dir, subfolder) if subfolder else out_dir
        os.makedirs(abs_dir, exist_ok=True)

        counter = 1
        while True:
            fname = f"{prefix}_{counter:05d}_.{ext}"
            fpath = os.path.join(abs_dir, fname)
            if not os.path.exists(fpath):
                break
            counter += 1

        body = text or ""
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(body)

        return {
            "ui": {
                "text": [body],
                "images": [{"filename": fname, "subfolder": subfolder, "type": "output"}],
            },
            "result": (body,),
        }


NODE_CLASS_MAPPINGS = {"VLObjects_SaveText": VLObjects_SaveText}
NODE_DISPLAY_NAME_MAPPINGS = {"VLObjects_SaveText": "VL Objects Save Text"}
