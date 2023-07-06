
process downloadModel {
    conda "requests conda-forge::tqdm"
    publishDir "$params.chkpt_dir", mode: 'copy'

    input:
    val model
    val model_type

    output:
    // This is where publishDir needs to come in, symlinking to chkpt repo
    path "${params.chkpt_fname}", emit: model_chkpt

    script:
    """
    download_model.py --chkpt-dir ${params.chkpt_dir} --model-name ${model} --model-type ${model_type}
    """
}

process runSAM {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_sam.yml"

    input:
    path image_path
    path model_config
    path model_chkpt
    val model_type
    
    output:
    // Switch this to use publishDir and avoid path manipulation in python
    stdout

    script:
    """
    run_sam.py --img-path ${image_path} --module-dir ${moduleDir} --model-chkpt ${model_chkpt} --model-type ${model_type} --model-config ${model_config}
    """
}

// process TESTSAM {
//     file('./nextflow/modules/models/envs/conda_sam.yml', checkIfExists: true)
//     file("${moduleDir}/envs/conda_sam.yml", checkIfExists: true)

//     output:
//     stdout

//     script:
//     """
//     echo ${moduleDir} ${projectDir} ${launchDir}
//     """
// }