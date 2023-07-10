import argparse
from pathlib import Path

import requests
from tqdm.auto import tqdm

SAM_MODELS = {
    "default": {
        "filename": "sam_default.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_h": {
        "filename": "sam_vit_h.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "filename": "sam_vit_l.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "filename": "sam_vit_b.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    "MedSAM": {
        "filename": "sam_MedSAM.pth",
        "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth",
    },
}

MODEL_BANK = {"sam": SAM_MODELS}


def get_model_checkpoint(chkpt_dir, model_name, model_type):
    # Get the model dict
    model_dict = MODEL_BANK[model_name]
    # Get the checkpoint filename
    chkpt_fname = Path(model_dict[model_type]["filename"])
    # Just return if this already exists
    # NOTE: Using chkpt_dir here as that's where Nextflow will copy the result to
    if (Path(chkpt_dir) / chkpt_fname).exists():
        return
    # Otherwise download it
    print(f"Downloading {model_name} ({model_type})")
    download_from_url(model_dict[model_type]["url"], chkpt_fname)


def download_from_url(url, chkpt_fname):
    # Open the URL and get the content length
    req = requests.get(url, stream=True)
    req.raise_for_status()
    content_length = int(req.headers.get("Content-Length"))

    # Download the file and update the progress bar
    with open(chkpt_fname, "wb") as f:
        with tqdm(
            desc=f"Downloading {chkpt_fname.name}...",
            total=content_length,
            unit="B",
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    # Close request
    req.close()
    print(f"Done! Checkpoint saved to {chkpt_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt-dir",
        required=True,
        type=str,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help="Model (name) selected by user",
    )
    parser.add_argument(
        "--model-type", required=True, type=str, help="Specific model variant"
    )

    args = parser.parse_args()

    get_model_checkpoint(
        chkpt_dir=args.chkpt_dir,
        model_name=args.model_name,
        model_type=args.model_type,
    )
