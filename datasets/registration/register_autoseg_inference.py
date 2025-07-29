# register_autoseg_inference.py
"""
Register a folder of images under the Detectron2 catalog for inference.
Dataset name: "autoseg_inference"
Place this file somewhere on your PYTHONPATH (e.g. in X_Decoder/datasets/registration),
and import it before building your dataloaders.
"""
import os
import glob
from detectron2.data import DatasetCatalog, MetadataCatalog

# Dataset name in Detectron2
DATASET_NAME = "autoseg_inference"

# File extensions to include
_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]


def _load_autoseg_inference_images():
    """
    Load image file paths from `input_dir` as a Detectron2 dataset.
    The `input_dir` is assumed to be specified via the environment variable
    AUTOG_SEG_INPUT_DIR, or falls back to project_root/AutoSeg/input.
    """
    # Determine input directory
    env_dir = os.getenv("AUTOG_SEG_INPUT_DIR")
    if env_dir:
        input_dir = os.path.abspath(env_dir)
    else:
        # two levels up: registration -> datasets -> X_Decoder; one more to project root
        cur = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(cur, '..', '..', '..'))
        input_dir = os.path.join(project_root, 'input')

    # Gather all image files
    image_paths = []
    for ext in _EXTS:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)

    # Build Detectron2-style dicts
    dataset_dicts = []
    for idx, file_name in enumerate(image_paths):
        record = {
            "file_name": file_name,
            "image_id": idx,
        }
        dataset_dicts.append(record)
    return dataset_dicts


# Register if not already present
if DATASET_NAME not in DatasetCatalog.list():
    DatasetCatalog.register(DATASET_NAME, _load_autoseg_inference_images)
    MetadataCatalog.get(DATASET_NAME).set(
        evaluator_type="inference",
    )
