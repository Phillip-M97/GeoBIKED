"""Storage for global variables."""

import os
from torch.cuda import is_available as cuda_available


MODEL_NAME_MAP_SHORT_TO_LONG = {
    "gpt-4": "gpt-4-vision-preview",
    "gpt-4o": "gpt-4o",
    "imp": "MILVLG/imp-v1-3b",
    "moondream": "vikhyatk/moondream2",
    "florence-base": "microsoft/Florence-2-base",
    "florence-base-ft": "microsoft/Florence-2-base-ft",
    "florence-large": "microsoft/Florence-2-large",
    "florence-large-ft": "microsoft/Florence-2-large-ft",
}
MODEL_NAME_MAP_LONG_TO_SHORT = {long: short for short, long in MODEL_NAME_MAP_SHORT_TO_LONG.items()}
DEVICE = "cuda" if cuda_available() else "cpu"

LENGTHS = ["short", "long"]
VIBES = ["technical", "casual"]
STYLES = ["marketing-message", "prompt-to-midjourney"]

DESCRIPTIONS_DATA_DIR = "..\\..\\descriptions"
DESCRIPTIONS_SCRATCHFILES_DIR = "..\\..\\descriptions\\scratchfiles"
IMAGES_DATA_DIR_256 = "..\\..\\images\\256x256"
IMAGES_DATA_DIR_2048 = "..\\..\\images\\2048x2048"
IMAGES_DATA_DIR_256_cropped = "..\\..\\images\\cropped_256x256"
IMAGES_DATA_DIR_512_cropped = "..\\..\\images\\cropped_512x512"
IMAGES_DATA_DIR_1024_cropped = "..\\..\\images\\cropped_1024x1024"
IMAGES_DATA_DIR_2048_cropped = "..\\..\\images\\cropped_2048x2048"

os.makedirs(DESCRIPTIONS_DATA_DIR, exist_ok=True)
os.makedirs(DESCRIPTIONS_SCRATCHFILES_DIR, exist_ok=True)
os.makedirs(IMAGES_DATA_DIR_256, exist_ok=True)
os.makedirs(IMAGES_DATA_DIR_2048, exist_ok=True)
os.makedirs(IMAGES_DATA_DIR_256_cropped, exist_ok=True)
os.makedirs(IMAGES_DATA_DIR_512_cropped, exist_ok=True)
os.makedirs(IMAGES_DATA_DIR_1024_cropped, exist_ok=True)
os.makedirs(IMAGES_DATA_DIR_2048_cropped, exist_ok=True)

IMAGE_SIZE_TO_DIRECTORY = {
    256: IMAGES_DATA_DIR_256_cropped,
    512: IMAGES_DATA_DIR_512_cropped,
    1024: IMAGES_DATA_DIR_1024_cropped,
    2048: IMAGES_DATA_DIR_2048_cropped,
}

COLUMNS_OF_INTEREST = [
    "bike_type",
    "has_bottle_seattube",
    "has_bottle_downtube",
    "rim_style_front",
    "rim_style_rear",
    "fork_type",
]


COLUMNS_OF_INTEREST_CONSOLIDATED_BOTTLES = [
    "bike_type",
    "rim_style_front",
    "rim_style_rear",
    "fork_type",
]
