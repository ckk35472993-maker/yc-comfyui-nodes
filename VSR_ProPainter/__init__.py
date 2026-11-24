from .vsr_propainter_node import (
    NODE_CLASS_MAPPINGS as _PP_C,
    NODE_DISPLAY_NAME_MAPPINGS as _PP_D,
)
from .vsr_lama_node import (
    NODE_CLASS_MAPPINGS as _LAMA_C,
    NODE_DISPLAY_NAME_MAPPINGS as _LAMA_D,
)
from .vsr_sttn_node import (
    NODE_CLASS_MAPPINGS as _STTN_C,
    NODE_DISPLAY_NAME_MAPPINGS as _STTN_D,
)
from .vsr_subtitle_detect_node import (
    NODE_CLASS_MAPPINGS as _DET_C,
    NODE_DISPLAY_NAME_MAPPINGS as _DET_D,
)
from .vsr_crop_paste_node import (
    NODE_CLASS_MAPPINGS as _CP_C,
    NODE_DISPLAY_NAME_MAPPINGS as _CP_D,
)

NODE_CLASS_MAPPINGS = {**_PP_C, **_LAMA_C, **_STTN_C, **_DET_C, **_CP_C}
NODE_DISPLAY_NAME_MAPPINGS = {**_PP_D, **_LAMA_D, **_STTN_D, **_DET_D, **_CP_D}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
