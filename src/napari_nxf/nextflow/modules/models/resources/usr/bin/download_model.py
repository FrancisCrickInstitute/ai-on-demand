import argparse
from pathlib import Path

import requests
from tqdm.auto import tqdm

SAM_MODELS = {
    "default": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    "MedSAM": {
        "filename": "sam_vit_b_01ec64_medsam.pth",
        "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth",
    },
}

MODEL_BANK = {"sam": SAM_MODELS}


def get_model_checkpoint(module_dir, model_name, model_type):
    """Check if the model checkpoint exists, and if not, download it"""
    # Get the module dir and make subfolder for checkpoints for this model
    module_dir = Path(module_dir)
    chkpt_path = module_dir / f"{model_name}_chkpts"
    chkpt_path.mkdir(parents=True, exist_ok=True)
    # Get the model dict
    model_dict = MODEL_BANK[model_name]
    # Get the checkpoint filename
    chkpt_fname = chkpt_path / model_dict[model_type]["filename"]
    # Just return if this already exists
    if chkpt_fname.exists():
        return
    # Otherwise download it
    print(f"Downloading {model_name} ({model_type})...")
    download_from_url(model_dict[model_type]["url"], chkpt_fname)


def download_from_url(url, chkpt_fname):
    # Open the URL and get the content length
    req = requests.get(url, stream=True)
    content_length = int(req.headers.get("Content-Length", 0))
    # Download the file and update the progress bar
    with open(chkpt_fname, "wb") as f:
        for chunk in tqdm(
            req.iter_content(chunk_size=1024 * 1024),
            total=content_length,
            unit="B",
            unit_scale=True,
        ):
            if chunk:
                f.write(chunk)
    # Close request
    req.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module-dir",
        required=True,
        type=str,
        help="Path to models module directory",
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
        module_dir=args.module_dir,
        model_name=args.model_name,
        model_type=args.model_type,
    )
