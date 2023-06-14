from pathlib import Path

import nextflow

nxf_params = {
    "img_dir": "/Users/shandc/Documents/ai-on-demand/src/napari_nxf/all_img_paths.txt",
    "model_config": "/Users/shandc/Documents/ai-on-demand/src/napari_nxf", 
    "model": "sam",
    "task": "everything",
    "executor": "local",
}

for execution in nextflow.run_and_poll(
    sleep=0.1,
    pipeline_path=str(Path(__file__).parent / "nextflow" / "main.nf"),
    run_path=Path(__file__).parent / "nextflow",
    params=nxf_params
):
    print(execution.command)
    print(f"status: {execution.status}  |  id: {execution.identifier}  | started: {execution.started}")
else:
    print(execution.log)