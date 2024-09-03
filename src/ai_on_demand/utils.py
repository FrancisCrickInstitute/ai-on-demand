import hashlib
import json
from napari.layers import Image
from napari.utils.notifications import show_info
from pathlib import Path
import textwrap
from typing import Optional, Union
import warnings
import yaml

from platformdirs import user_cache_dir


def sanitise_name(name: str) -> str:
    """
    Function to sanitise model/model variant names to use in filenames (in Nextflow).
    """
    return name.replace(" ", "-")


def merge_dicts(d1: dict, d2: Optional[dict] = None) -> dict:
    """
    Merge two dictionaries recursively. d2 will overwrite d1 where specified.

    Assumes both dicts have same structure/keys.
    """
    # Short-circuit if d2 is None
    if d2 is None:
        return d1
    # Otherwise recursively merge
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1


def format_tooltip(text: str, width: int = 70) -> str:
    """
    Function to wrap text in a tooltip to the specified width. Ensures better-looking tooltips.

    Necessary because Qt only automatically wordwraps rich text, which has it's own issues.
    """
    return textwrap.fill(text.strip(), width=width, drop_whitespace=True)


def filter_empty_dict(d: dict) -> dict:
    """
    Filter out empty dicts from a nested dict.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = filter_empty_dict(v)
        if v not in (None, {}):
            new_dict[k] = v
    return new_dict


def calc_param_hash(d: dict) -> str:
    # Sort the dictionary so that the hash is consistent on contents rather than order
    sorted_d = dict(sorted(d.items()))
    return hashlib.md5(json.dumps(sorted_d).encode("utf-8")).hexdigest()


def load_config(config_path: Union[str, Path]) -> dict:
    with open(Path(config_path), "r") as f:
        if config_path.suffix == ".json":
            model_dict = json.load(f)
        elif config_path.suffix in (".yaml", ".yml"):
            model_dict = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Config file (path: {config_path}) is not JSON or YAML!"
            )
    return model_dict


def get_plugin_cache() -> tuple[Path, Path]:
    cache_dir = Path(user_cache_dir("aiod"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings_path = cache_dir / "aiod_settings.yaml"
    return cache_dir, settings_path


def load_settings() -> dict:
    _, settings_path = get_plugin_cache()

    if settings_path.exists():
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
    else:
        settings = {}
    return settings


def get_image_layer_path(
    img_layer: Image, image_path_dict: Optional[dict] = None
) -> Path:
    # Skip this if the layer is a result of the Preprocess preview
    if img_layer.metadata.get("preprocess", None):
        return
    # Extract from the layer source
    img_path = img_layer.source.path
    # If not there, check the metadata
    # This occurs explicitly with the sample data by design (because I have to)
    if img_path is None:
        try:
            img_path = img_layer.metadata["path"]
        except KeyError:
            img_path = None
    # If still None, check if already added
    if img_path is None:
        if image_path_dict is not None:
            if img_layer.name not in image_path_dict:
                show_info(
                    f"Cannot extract path for image layer {img_layer}. Please add manually using the buttons."
                )
                return
    else:
        return Path(img_path)


def get_img_dims(
    layer: Image, img_path: Optional[Path] = None
) -> tuple[int, int, int, Optional[int]]:
    # Squeeze the data to remove any singleton dimensions
    arr = layer.data.squeeze()
    # TODO: What if multi-channel but not RGB? Does Napari allow this?
    # Check if the image is RGB or not
    if layer.rgb:
        res = arr.shape[:-1]
        channels = arr.shape[-1]
    else:
        res = arr.shape
        channels = None
    # It could be multi-channel but not RGB
    # 2D
    if len(res) == 2:
        num_slices = 1
        H, W = res
        channels = 1 if channels is None else channels
    # 3D
    elif len(res) == 3:
        # Without metadata, we can't know if the 3rd dimension is channels or slices
        # TODO: Use bioio to get metadata and infer this
        num_slices, H, W = res
        channels = 1 if channels is None else channels
        warnings.warn(
            f"Assuming the first dimension is slices for {layer.name} image layer ({layer} with shape {res})."
        )
    # 4D
    elif len(res) == 4:
        # We assume the first two dims are slices and channels, in some order
        # Assume whichever is smaller are the channels
        warnings.warn(
            f"Assuming the first two dimensions are channels and slices for {layer.name} image layer ({layer}), and using the smaller of the two as the number of channels."
        )
        if channels is not None:
            num_slices, H, W = res
        else:
            if res[0] < res[1]:
                channels, num_slices, H, W = res
            else:
                num_slices, channels, H, W = res
    # Who knows
    else:
        if img_path is None:
            raise ValueError(
                f"Unexpected number of dimensions for {layer.name} image layer ({layer})!"
            )
        else:
            raise ValueError(
                f"Unexpected number of dimensions for image {img_path}!"
            )
    return H, W, num_slices, channels
