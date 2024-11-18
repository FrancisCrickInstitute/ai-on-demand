from pathlib import Path
from typing import Union

import aiod_utils.rle


def get_rle_reader(path: Union[str, list[str]]):
    # If the path is a list, take the first element to get the extension
    if isinstance(path, list):
        path = path[0]
    path = Path(path)
    # Return our reader if the extension is in the accepted extensions
    return rle_reader if path.suffix in aiod_utils.rle.EXTENSIONS else None


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
        mask, metadata = aiod_utils.rle.decode(encoded_mask, mask_type)
        # Flatten metadata if needed
        if "metadata" in metadata:
            metadata = metadata["metadata"]
        # TODO: Handle scale metadata if given for downsampled masks
        layer_tuples.append(
            (
                mask,
                {
                    "name": Path(path).stem,
                    "metadata": {
                        "path": path,
                        "mask_type": mask_type,
                        **metadata,
                    },
                },
                "labels",
            )
        )
    return layer_tuples


def rle_writer(path: str, data, attributes: dict) -> list[str]:
    # Check the suffix and add if necessary
    # NOTE: Not sure if needed and what Napari handles
    suffix = Path(path).suffix
    if suffix == "":
        path = Path(path).with_suffix(".rle")
    elif suffix not in aiod_utils.rle.EXTENSIONS:
        return None
    # Encode the data
    # TODO: Get the attributes and insert as metadata into the RLE
    metadata = attributes.get("metadata", {})
    encoded_mask = aiod_utils.rle.encode(data, metadata=metadata)
    # Save the encoded mask
    aiod_utils.rle.save_encoding(encoded_mask, path)
    return [path]
