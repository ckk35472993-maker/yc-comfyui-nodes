import os
from backend.config import config, BASE_DIR
from backend.tools.common_tools import merge_big_file_if_not_exists
from backend.tools.constant import SubtitleDetectMode

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER: "PP-OCRv5_server_det",
}

class ModelConfig:
    def __init__(self):
        # 本地桥接补丁：VSR_MODELS_ROOT 指向 VideoAI_Data 模型目录（默认走 VSR 自带 models/）
        _mroot = os.environ.get("VSR_MODELS_ROOT", "").strip() or os.path.join(BASE_DIR, 'models')
        # 本地桥接补丁：VSR_LAMA_MODEL_DIR 允许复用已有 big-lama 目录（modules/models/big-lama）
        self.LAMA_MODEL_DIR = os.environ.get("VSR_LAMA_MODEL_DIR", "").strip() or os.path.join(_mroot, 'big-lama')
        self.STTN_AUTO_MODEL_PATH = os.path.join(_mroot, 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(_mroot, 'sttn-det', 'sttn.pth')
        self.PROPAINTER_MODEL_DIR = os.path.join(_mroot, 'propainter')
        if config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE:
            self.DET_MODEL_DIR = os.path.join(_mroot, 'V5', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_SERVER:
            self.DET_MODEL_DIR = os.path.join(_mroot, 'V5', 'ch_det')
        if config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR,'models', 'V5', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_SERVER:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'V5', 'ch_det')
        else:
            raise ValueError(f"Invalid subtitle detect mode: {config.subtitleDetectMode.value}")
        self.DET_MODEL_NAME = _MODEL_NAME_MAP[config.subtitleDetectMode.value]

        merge_big_file_if_not_exists(self.LAMA_MODEL_DIR, 'bit-lama.pt')
        merge_big_file_if_not_exists(self.PROPAINTER_MODEL_DIR, 'ProPainter.pth')
