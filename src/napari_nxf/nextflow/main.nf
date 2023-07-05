#!/usr/bin/env nextflow
nextflow.enable.dsl=2

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

    // Create channel from paths to each image file
    img_ch = Channel.fromPath( params.img_dir )
                    .splitText()

    // Download model checkpoint
    downloadModel( params.model, params.model_type )
    // Select appropriate model
    if( params.model == "sam" )
        runSAM(img_ch, params.model_config, downloadModel.out.model_chkpt, params.model_type)
    else
        error "Model ${params.model} not yet implemented!"
}

workflow.onComplete{
    log.info ( workflow.success ? '\nDone!' : '\nSomething went wrong!' )
}