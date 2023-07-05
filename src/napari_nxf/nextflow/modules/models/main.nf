
process downloadModel {
    conda "requests conda-forge::tqdm"

    input:
    val model
    val model_type

    output:
    // This is where publishDir needs to come in, symlinking to chkpt repo
    path "${moduleDir}/${model}_chkpts/${model}_${model_type}.pth", emit: model_chkpt

    script:
    """
    download_model.py --module-dir ${moduleDir} --model-name ${model} --model-type ${model_type}
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