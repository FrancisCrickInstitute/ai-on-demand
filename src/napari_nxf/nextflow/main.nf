#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.chkpt_dir = "${projectDir}/modules/models/${params.model}_chkpts"
params.chkpt_fname = "${params.model}_${params.model_type}.pth"
params.chkpt_path = "${params.chkpt_dir}/${params.chkpt_fname}"

include { downloadModel; runSAM } from './modules/models'

log.info """\
         AI ON DEMAND PIPELINE
         ===========================
         Model Name     : ${params.model}
         Model Variant  : ${params.model_type}
         Task           : ${params.task}
         Model config   : ${params.model_config}
         Image filepaths: ${params.img_dir}
         Executor       : ${params.executor}
         """.stripIndent()

workflow {
    // TODO: Move the model-based stuff into a workflow under the models module
    // Download model checkpoint if it doesn't exist
    chkpt_fname = file( params.chkpt_path )

    if ( !chkpt_fname.exists() ) {
        downloadModel( params.model, params.model_type )
    }

    // Create channel from paths to each image file
    img_ch = Channel.fromPath( params.img_dir )
                    .splitText()

    // Select appropriate model
    if( params.model == "sam" )
        runSAM(img_ch, params.model_config, params.chkpt_path, params.model_type)
    else
        error "Model ${params.model} not yet implemented!"
}

workflow.onComplete{
    log.info ( workflow.success ? '\nDone!' : '\nSomething went wrong!' )
}