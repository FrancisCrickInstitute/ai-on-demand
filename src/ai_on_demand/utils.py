import hashlib
import json
from napari.layers import Image
from napari.utils.notifications import show_info
from pathlib import Path
import textwrap
from typing import Optional, Union
import warnings
import yaml
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QDialog,
    QTextEdit,
)
from platformdirs import user_cache_dir
import aiod_utils.io


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
    return textwrap.fill(
        text.strip(),
        width=width,
        drop_whitespace=True,
        replace_whitespace=True,
    )


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


def load_model_config(config_path: Union[str, Path]) -> dict:
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
    layer: Image, img_path: Optional[Path] = None, verbose: bool = True
) -> tuple[int, int, int, Optional[int]]:
    # Hope image loaded with custom bioio loader, or that the original file can be read
    dims = (
        layer.metadata.get("dimensions")
        or aiod_utils.io.load_image(get_image_layer_path(layer)).dims
    )
    # TODO: allow time dimension instead of Z for some models
    # TODO: explicitly check for multi-channel RGB image
    return (
        dims.Y,
        dims.X,
        dims.Z,
        dims.S if layer.rgb or "S" in dims.order else dims.C,
    )
    # FIXME: catch issues with finding dims and extract directly from layer data as fallback


class InfoWindow(QDialog):
    def __init__(self, parent=None, title: str = "", content: str = ""):
        super().__init__(parent)

        # Set the layout
        self.layout = QVBoxLayout()
        # Set the window title
        self.setWindowTitle(title)
        # Add the info label
        self.info_label = QTextEdit()
        # Make the text selectable, but not editable
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setText(content)
        self.info_label.setMinimumSize(500, 500)

        self.layout.addWidget(self.info_label)
        self.setLayout(self.layout)