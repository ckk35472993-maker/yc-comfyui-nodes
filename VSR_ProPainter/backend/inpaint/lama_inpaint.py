import os
import gc
from typing import Union, List
import torch
import numpy as np
from PIL import Image
from backend.inpaint.utils.lama_util import prepare_img_and_mask, get_image, pad_img_to_modulo
from backend import config
from backend.tools.inpaint_tools import get_inpaint_area_by_mask

# 大图保护阈值：LaMa FFC 在高分辨率上显存/索引爆炸（14K 图单次申请 26GiB，
# 或触发 "input tensor must fit into 32-bit index math"）。仅对 mask 周边有限窗口
# 推理，再把窗口结果贴回原分辨率——mask 外像素零损失。
LAMA_MAX_INPAINT_SIDE = int(os.environ.get("LAMA_MAX_INPAINT_SIDE", "2048"))
# mask bbox 外扩上下文比例（相对 bbox 长边），保证 LaMa 有足够背景信息可用
LAMA_MASK_CONTEXT_RATIO = float(os.environ.get("LAMA_MASK_CONTEXT_RATIO", "0.6"))


class LamaInpaint:
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path='big-lama.pt') -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

    def _run_model(self, img_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """对给定尺寸的图+mask 跑一次 LaMa 推理，返回裁剪回原尺寸的 uint8 结果。"""
        h, w = img_np.shape[:2]
        image, mask = prepare_img_and_mask(img_np, mask_np, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            return cur_res[:h, :w]

    def inpaint(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]):
        img_np = image if isinstance(image, np.ndarray) else np.array(image)
        mask_np = mask if isinstance(mask, np.ndarray) else np.array(mask)
        H, W = img_np.shape[:2]

        mask_bin = mask_np[..., 0] if mask_np.ndim == 3 else mask_np
        ys, xs = np.where(mask_bin > 0)
        if ys.size == 0:
            # 空 mask：无区域需要修复，直接返回原图（避免大图白跑一次推理/OOM）
            return img_np.copy()

        my1, my2 = int(ys.min()), int(ys.max()) + 1
        mx1, mx2 = int(xs.min()), int(xs.max()) + 1
        bw, bh = mx2 - mx1, my2 - my1
        pad = int(max(bw, bh) * LAMA_MASK_CONTEXT_RATIO)
        x1, y1 = max(0, mx1 - pad), max(0, my1 - pad)
        x2, y2 = min(W, mx2 + pad), min(H, my2 + pad)
        cw, ch = x2 - x1, y2 - y1

        # 整图未超限时维持原行为（小图全图推理，LaMa 利用全图上下文效果更好）
        if max(H, W) <= LAMA_MAX_INPAINT_SIDE and max(cw, ch) <= LAMA_MAX_INPAINT_SIDE:
            return self._run_model(img_np, mask_np)

        # 大图：仅对 mask 周边窗口推理；窗口仍超限（mask 巨大）时等比缩到阈值内
        sub_img = img_np[y1:y2, x1:x2]
        sub_mask = mask_np[y1:y2, x1:x2]
        scale = min(1.0, LAMA_MAX_INPAINT_SIDE / max(cw, ch))
        if scale < 1.0:
            import cv2
            tw, th = max(8, int(cw * scale)), max(8, int(ch * scale))
            sub_res = self._run_model(
                cv2.resize(sub_img, (tw, th), interpolation=cv2.INTER_AREA),
                cv2.resize(sub_mask, (tw, th), interpolation=cv2.INTER_NEAREST),
            )
            sub_res = cv2.resize(sub_res, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        else:
            sub_res = self._run_model(sub_img, sub_mask)
        # 只把 mask 覆盖区贴回（膨胀+羽化过渡），窗口内其余像素保持原图不损画质
        import cv2
        win_mask = (mask_bin[y1:y2, x1:x2] > 0).astype(np.uint8)
        win_mask = cv2.dilate(win_mask, np.ones((9, 9), np.uint8), iterations=1)
        alpha = cv2.GaussianBlur(win_mask.astype(np.float32), (9, 9), 0)[..., None]
        blended = sub_res.astype(np.float32) * alpha + sub_img.astype(np.float32) * (1.0 - alpha)
        out = img_np.copy()
        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def _inpaint_batch(self, images: List[np.ndarray], masks: List[np.ndarray]):
        """批量推理：将多帧分小批次送入 GPU，避免单次推理过大导致卡死"""
        if len(images) == 1:
            return [self.inpaint(images[0], masks[0])]

        orig_height, orig_width = images[0].shape[:2]
        # 分小批次推理，每批最多 4 帧
        mini_batch_size = 4
        results = [None] * len(images)
        for start in range(0, len(images), mini_batch_size):
            end = min(start + mini_batch_size, len(images))
            batch_imgs = []
            batch_masks = []
            for i in range(start, end):
                batch_imgs.append(get_image(images[i]))
                batch_masks.append(get_image(masks[i]))

            padded_imgs = np.stack([pad_img_to_modulo(img, 8) for img in batch_imgs])
            padded_masks = np.stack([pad_img_to_modulo(m, 8) for m in batch_masks])

            img_tensor = torch.from_numpy(padded_imgs).to(self.device)
            mask_tensor = torch.from_numpy(padded_masks).to(self.device)
            mask_tensor = (mask_tensor > 0) * 1

            with torch.inference_mode():
                inpainted = self.model(img_tensor, mask_tensor)
                batch_results = inpainted.permute(0, 2, 3, 1).detach().cpu().numpy()
                batch_results = np.clip(batch_results * 255, 0, 255).astype('uint8')

            for i in range(end - start):
                results[start + i] = batch_results[i][:orig_height, :orig_width]

            del img_tensor, mask_tensor, padded_imgs, padded_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param input_mask: 字幕区域mask
        """
        mask = input_mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, mask)
        # 高分辨率帧存储列表
        frames_hr = [f.copy() for f in input_frames]
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []

        for k in range(len(inpaint_area)):
            # 收集该区域的所有裁剪帧和遮罩
            cropped_frames = []
            cropped_masks = []
            for j in range(len(frames_hr)):
                image_crop = frames_hr[j][inpaint_area[k][0]:inpaint_area[k][1], :, :]
                mask_crop = mask[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                cropped_frames.append(image_crop)
                cropped_masks.append(mask_crop)

            # 批量推理
            comps[k] = self._inpaint_batch(cropped_frames, cropped_masks)
            del cropped_frames, cropped_masks
            gc.collect()

        # 如果存在去除部分
        if inpaint_area:
            import cv2
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    y1, y2 = inpaint_area[k][0], min(inpaint_area[k][1], H_ori)
                    if y2 - y1 <= 0:
                        continue
                    comp = comps[k][j]
                    # 只把 mask 覆盖区贴回（与 inpaint() 窗口同款 9×9 膨胀+羽化过渡），
                    # 条带内其余像素保留原帧，避免整带被 LaMa 解码器重渲染损质
                    win_mask = (mask[y1:y2, :] > 0).astype(np.uint8)
                    win_mask = cv2.dilate(win_mask, np.ones((9, 9), np.uint8), iterations=1)
                    alpha = cv2.GaussianBlur(win_mask.astype(np.float32), (9, 9), 0)
                    if alpha.ndim == 2:
                        alpha = alpha[..., None]
                    blended = comp.astype(np.float32) * alpha + frame[y1:y2, :, :].astype(np.float32) * (1.0 - alpha)
                    frame[y1:y2, :, :] = np.clip(blended, 0, 255).astype(np.uint8)
                inpainted_frames.append(frame)
        else:
            # 无需处理的区域，返回原始帧
            inpainted_frames = frames_hr

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return inpainted_frames


