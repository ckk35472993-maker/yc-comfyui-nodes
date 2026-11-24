"""ComfyUI-YC-VideoKit —— MiniMax H3 视频链路节点合集（单包安装）。

包含两组节点：

- ``MiniMaxH3PlanV2*`` 计划链节点：ProjectSetup / Image&Video&Audio Reference /
  Shot / CharacterReplacement / SubjectBinding / DialogueEvent / FoleyTarget /
  ShotKeyframe / ShotMotionReference / PromptMerge / PromptReview /
  PromptOverride / ApplyReferencePlan。把「分镜+参考素材+台词时序」编译为
  合法 H3 prose，并路由进原生 I2V/R2V 采样路径。
- 遮罩/媒体辅助节点：SelectObjectMask（SAM3 多目标按空间语义选一）、
  TemporalUnionMask / MaskGridPreview / PerRowMaskPatch / SetGenerationMask
  （generation-mask 补丁）、TrimSourceAV（源片音视频定长裁剪）。

运行零三方依赖（numpy/torch 由 ComfyUI 环境提供）。
"""

from .h3_plan_nodes import (
    NODE_CLASS_MAPPINGS as PLAN_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PLAN_NODE_DISPLAY_NAME_MAPPINGS,
)
from .h3_review import (
    NODE_CLASS_MAPPINGS as REVIEW_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as REVIEW_NODE_DISPLAY_NAME_MAPPINGS,
)
from .h3_plan_apply import (
    NODE_CLASS_MAPPINGS as APPLY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as APPLY_NODE_DISPLAY_NAME_MAPPINGS,
)
from .mask_nodes import (
    NODE_CLASS_MAPPINGS as MASK_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_NODE_DISPLAY_NAME_MAPPINGS,
)

# h3_review_routes 注册 aiohttp 路由，仅 ComfyUI 服务进程内可用；
# 裸 python 导入（离线校验等）无 PromptServer——失败不阻断节点注册。
try:
    from . import h3_review_routes as _h3_review_routes  # noqa: F401
except Exception:
    _h3_review_routes = None

NODE_CLASS_MAPPINGS = {
    **PLAN_NODE_CLASS_MAPPINGS,
    **REVIEW_NODE_CLASS_MAPPINGS,
    **APPLY_NODE_CLASS_MAPPINGS,
    **MASK_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PLAN_NODE_DISPLAY_NAME_MAPPINGS,
    **REVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    **APPLY_NODE_DISPLAY_NAME_MAPPINGS,
    **MASK_NODE_DISPLAY_NAME_MAPPINGS,
}

# 双名注册：MiniMaxH3* 原名（兼容存量工作流）+ YC* 别名（对外品牌工作流），
# 两套名字指向同一类实现，ComfyUI 按 class_type 名字查找，互不干扰。
_ALIAS = {}
for _name, _cls in list(NODE_CLASS_MAPPINGS.items()):
    if _name.startswith("MiniMaxH3"):
        _alias = "YC" + _name[len("MiniMaxH3"):]
        if _alias not in NODE_CLASS_MAPPINGS:
            _ALIAS[_alias] = _cls
            NODE_DISPLAY_NAME_MAPPINGS.setdefault(
                _alias, NODE_DISPLAY_NAME_MAPPINGS.get(_name, _alias)
            )
NODE_CLASS_MAPPINGS.update(_ALIAS)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
