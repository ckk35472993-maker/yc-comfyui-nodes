import argparse
from enum import Enum

from .constant import InpaintMode

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    parser.add_argument(
        "--mask", "-m", type=str, default=None,
        help="External mask image path (grayscale). When provided, skip internal subtitle detection."
    )
    parser.add_argument(
        "--mask-video", type=str, default=None,
        help="Per-frame mask video (grayscale mp4). Each frame's mask is used for the corresponding source frame. "
             "When provided, overrides --mask and enables per-frame dynamic inpainting with propainter/lama."
    )
    parser.add_argument(
        "--no-detect", action="store_true",
        help="Skip subtitle detection; use --subtitle-area-coords or --mask directly"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Do not merge original audio into output video"
    )
    parser.add_argument(
        "--lang", type=str, default="ch",
        choices=["ch", "japan", "korean", "latin", "chinese_cht", "en", "ko", "es", "vi"],
        help="OCR language for subtitle detection (default: ch). Used when --no-detect is not set."
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    return args