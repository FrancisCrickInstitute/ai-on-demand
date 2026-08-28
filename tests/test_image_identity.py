"""
Tests for keying per-image state by image_id rather than by filename stem.

Two images can share a stem (cells.tif and cells.png), and napari renames
duplicate layers ("cells", "cells [1]"), so neither the stem nor the layer name
is a usable identity. These pin the two halves of the fix: image_key for the
dicts, and path-based layer resolution instead of name lookups.
"""

import types
from pathlib import Path

import numpy as np
import pytest
from napari.layers import Image

from aiod_napari.utils import (
    find_image_layer,
    get_image_layer_path,
    image_key,
    require_image_layer,
)


def make_layer(name, path=None, metadata_path=None):
    """An Image layer with the path recorded the way napari or our reader would."""
    layer = Image(np.zeros((4, 4)), name=name)
    if metadata_path is not None:
        layer.metadata["path"] = metadata_path
    if path is not None:
        layer._source = types.SimpleNamespace(path=str(path))
    return layer


class TestImageKey:
    def test_same_stem_different_extension_are_distinct(self):
        assert image_key(Path("/data/cells.tif")) != image_key(Path("/data/cells.png"))

    def test_key_ignores_the_parent_directory(self):
        # Deliberate: these two genuinely cannot be told apart downstream, so
        # they must collide here where the widget can warn about it
        assert image_key(Path("/a/cells.tif")) == image_key(Path("/b/cells.tif"))

    def test_key_matches_the_segment_flow_image_id(self):
        from aiod_utils.io import get_image_id

        path = Path("/data/cells.ome.tiff")
        assert image_key(path) == get_image_id(path).value

    def test_compound_extension_not_left_on_the_key(self):
        assert image_key(Path("/data/cells.ome.tiff")) == "cells_ome_tiff"


class TestFindImageLayer:
    def test_finds_by_source_path(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        path = Path("/data/cells.tif")
        viewer.add_layer(make_layer("cells", path=path))
        assert find_image_layer(viewer, path) is not None

    def test_finds_by_metadata_path_for_sample_data(self, make_napari_viewer_proxy):
        # Sample data records its path in metadata only, with no layer source
        viewer = make_napari_viewer_proxy()
        path = Path("/data/cells.tif")
        viewer.add_layer(make_layer("cells", metadata_path=path))
        assert find_image_layer(viewer, path) is not None

    def test_distinguishes_same_stem_images_napari_renamed(
        self, make_napari_viewer_proxy
    ):
        # The failure a name lookup cannot avoid: napari makes the second layer
        # "cells [1]", so viewer.layers["cells"] silently returns the first
        viewer = make_napari_viewer_proxy()
        tif, png = Path("/data/cells.tif"), Path("/data/cells.png")
        viewer.add_layer(make_layer("cells", path=tif))
        viewer.add_layer(make_layer("cells", path=png))
        assert find_image_layer(viewer, tif) is not find_image_layer(viewer, png)
        assert get_image_layer_path(find_image_layer(viewer, png)) == png

    def test_survives_a_renamed_layer(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        path = Path("/data/cells.tif")
        layer = make_layer("cells", path=path)
        viewer.add_layer(layer)
        layer.name = "whatever the user typed"
        assert find_image_layer(viewer, path) is layer

    def test_returns_none_when_absent(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        assert find_image_layer(viewer, Path("/data/missing.tif")) is None

    def test_ignores_preprocess_preview_layers(self, make_napari_viewer_proxy):
        # get_image_layer_path deliberately skips preview layers, so a preview
        # of an image must not be mistaken for the image itself
        viewer = make_napari_viewer_proxy()
        path = Path("/data/cells.tif")
        preview = make_layer("cells preview", path=path)
        preview.metadata["preprocess"] = True
        viewer.add_layer(preview)
        assert find_image_layer(viewer, path) is None

    def test_require_raises_with_the_path_in_the_message(
        self, make_napari_viewer_proxy
    ):
        viewer = make_napari_viewer_proxy()
        with pytest.raises(ValueError, match="missing.tif"):
            require_image_layer(viewer, Path("/data/missing.tif"))


class TestDataWidgetRegistration:
    @pytest.fixture
    def data_widget(self, make_napari_viewer_proxy):
        # DataWidget alone, so these do not depend on the Segment-Flow
        # submodule being checked out (which the full Inference widget needs)
        from aiod_napari.inference.data_selection import DataWidget

        return DataWidget(make_napari_viewer_proxy())

    def test_same_stem_different_extension_both_kept(self, data_widget):
        tif, png = Path("/data/cells.tif"), Path("/data/cells.png")
        assert data_widget._register_path(tif)
        assert data_widget._register_path(png)
        assert set(data_widget.image_path_dict.values()) == {tif, png}

    def test_same_name_across_directories_is_refused(self, data_widget):
        # Nothing downstream can separate these, so reject the second rather
        # than overwrite the first and silently run on one image
        first, second = Path("/a/cells.tif"), Path("/b/cells.tif")
        assert data_widget._register_path(first)
        assert not data_widget._register_path(second)
        assert list(data_widget.image_path_dict.values()) == [first]

    def test_re_registering_the_same_path_is_idempotent(self, data_widget):
        path = Path("/data/cells.tif")
        assert data_widget._register_path(path)
        assert data_widget._register_path(path)
        assert list(data_widget.image_path_dict.values()) == [path]

    def test_removal_uses_the_same_key(self, data_widget, make_napari_viewer_proxy):
        # on_layer_removed has to derive the key the same way registration did,
        # or entries leak and the run includes images no longer in the viewer
        tif, png = Path("/data/cells.tif"), Path("/data/cells.png")
        data_widget._register_path(tif)
        data_widget._register_path(png)
        layer = make_layer("cells", path=png)
        data_widget.on_layer_removed(types.SimpleNamespace(value=layer))
        assert list(data_widget.image_path_dict.values()) == [tif]

    def test_pathless_layers_are_skipped(self, make_napari_viewer_proxy):
        # An array added straight to the viewer has no file behind it, so there
        # is no image_id to key it by - it must be skipped, not crash the widget
        from aiod_napari.inference.data_selection import DataWidget

        viewer = make_napari_viewer_proxy()
        viewer.add_layer(make_layer("just an array"))
        widget = DataWidget(viewer)
        assert widget.image_path_dict == {}

    def test_removal_of_sample_data_layer(self, data_widget):
        # Path in metadata only - previously handled by a layer-name fallback
        path = Path("/data/cells.tif")
        data_widget._register_path(path)
        layer = make_layer("cells", metadata_path=path)
        data_widget.on_layer_removed(types.SimpleNamespace(value=layer))
        assert data_widget.image_path_dict == {}
