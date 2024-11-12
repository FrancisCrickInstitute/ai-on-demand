from pathlib import Path
from typing import Union

import aiod_utils.rle


def get_rle_reader(path: Union[str, list[str]]):
    # If the path is a list, take the first element to get the extension
    if isinstance(path, list):
        path = path[0]
    path = Path(path)
    # Accepted extensions
    # NOTE: It should only ever really be the first as that's what aiod_utils.rle.save creates
    extensions = (".pkl", ".pickle", ".rle")
    # Return our reader if the extension is in the accepted extensions
    return rle_reader if path.suffix in extensions else None


def rle_reader(paths: Union[str, list[str]]):
    if not isinstance(paths, list):
        paths = [paths]
    # Container for Napari layers
    layer_tuples = []
    # Loop over each given file
    for path in paths:
        # Load & decode the RLE
        encoded_mask = aiod_utils.rle.load_encoding(path)
        # NOTE: Only doing this to insert type as metadata
        mask_type = aiod_utils.rle.check_rle_type(encoded_mask)
        mask = aiod_utils.rle.decode(encoded_mask, mask_type)
        layer_tuples.append(
            (
                mask,
                {
                    "name": Path(path).stem,
                    "metadata": {"path": path, "mask_type": mask_type},
                },
                "labels",
            )
        )
    return layer_tuples
