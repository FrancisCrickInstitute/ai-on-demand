from pathlib import Path
from typing import Union
from functools import partial
import numpy as np
from bioio import BioImage
from bioio_base.reader import Reader

import aiod_utils.rle
import aiod_utils.io


def get_bioio_reader(path: Union[str, Path]):
    # Check if bioio can read this
    try:
        plugin = BioImage.determine_plugin(path)
        # If determine_plugin returns None, no plugin can handle this file
        if plugin is None:
            return None
        # NOTE: this reduces redundancy, as BioImage.__init__() internally will call determine_plugin() again internally anyway, unless a specific reader is forwarded to BioImage later on.
        return partial(
            bioio_reader, bioio_reader_class=plugin.metadata.get_reader()
        )
    except (
        AttributeError,
        FileNotFoundError,
    ):
        return None


def bioio_reader(
    path: Union[str, Path], bioio_reader_class: Union[Reader, None]
):
    # Load the image with utils loader, keeping defaults
    path = Path(path)
    bioio_img = aiod_utils.io.load_image(
        path,
        reader=bioio_reader_class,
    )
    return prepare_bioio_as_napari_layer(bioio_img, path)


def prepare_bioio_as_napari_layer(bioio_img, path):
    # Return LaterData tuple
    # Handle JPEGs etc.
    if isinstance(bioio_img, np.ndarray):
        return [(bioio_img, {"name": path.stem}, "image")]
    # Otherwise deal with BioIO object
    # TODO: Add metadata?
    # TODO: We specify where we want the channel axis to be, so can pass this to the viewer
    # Though if this splits one file to multiple layers...well, that needs addressing anyway
    try:
        bioio_metadata = bioio_img.ome_metadata
    except NotImplementedError:
        bioio_metadata = bioio_img.metadata
    # NOTE: https://github.com/bioio-devs/bioio/issues/25 issue for adding units
    try:
        pixel_sizes = bioio_img.physical_pixel_sizes
    except NotImplementedError:
        pixel_sizes = None
    # Add various metadata to the image
    metadata = {
        "bioio_metadata": bioio_metadata,
        "pixel_sizes": pixel_sizes,
        "bioio_dims": bioio_img.dims,
    }
    return [
        (
            bioio_img.get_image_dask_data(
                dimension_order_out="CZYX"
            ).squeeze(),
            {"name": path.stem, "metadata": metadata},
            "image",
        )
    ]


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
