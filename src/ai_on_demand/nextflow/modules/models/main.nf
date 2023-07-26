
process downloadModel {
    conda "${moduleDir}/envs/conda_download_model.yml"
    publishDir "$params.chkpt_dir", mode: 'copy'

    input:
    val model
    val model_type

    output:
    // This is where publishDir needs to come in, symlinking to chkpt repo
    path "${params.chkpt_fname}", emit: model_chkpt

    script:
    """
    python ${moduleDir}/resources/usr/bin/download_model.py --chkpt-dir ${params.chkpt_dir} --model-name ${model} --model-type ${model_type}
    """
}

process runSAM {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_sam.yml"
    // Switch this to use publishDir and avoid path manipulation in python?

    input:
    tuple path(image_path), val(mask_fname)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    // Because we are manually saving it in the .cache so napari can watch for each slice
    stdout

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_sam.py --img-path ${image_path} --mask-fname ${mask_fname} --output-dir ${mask_output_dir} --model-chkpt ${model_chkpt} --model-type ${model_type} --model-config ${model_config}
    """
}
