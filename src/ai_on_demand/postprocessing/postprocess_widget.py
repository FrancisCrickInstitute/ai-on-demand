import napari

from ai_on_demand.widget_classes import MainWidget, SubWidget


class Postprocess(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer=napari_viewer,
            title="Postprocess Masks",
            tooltip="""
Postprocess masks using various methods. This includes merging, splitting, and filtering masks.
""",
        )


class MergeMasks(SubWidget):
    """
    1. Union of two sets
    2. Parameterised vote of multiple sets
    3. Visualization of multiple sets
    4. Merge of specific labels
    5. Merge of labels based on distance (2D or 3D)
    6. Merge of labels based on overlap (3D)
    """

    pass


class FilterMasks(SubWidget):
    """
    1. Filter by size
    2. Filter by shape (or anything in regionprops?)
    3. Filter by label (i.e. delete)
    4. Filter boundary labels
    """

    pass


class MorphMasks(SubWidget):
    """
    1. Dilate, erode, open, close
    2. Fill holes
    """

    pass


class ExportMasks(SubWidget):
    """
    1. Export bounding boxes?
    2. Export masks as RLE
    3. Export masks as images
    4. Export as binary
    5. Export as labelled
    """

    pass
