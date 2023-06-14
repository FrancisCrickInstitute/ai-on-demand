#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { RUNSAM } from './modules/models'

// params.model = "sam"
// params.task = "everything"
// params.model_config = "/Users/shandc/Documents/ai-on-demand/src/napari_nxf"
// params.img_dir = "/Users/shandc/Documents/ai-on-demand/src/napari_nxf/all_img_paths.txt"
// params.executor = "Local"

log.info """\
         AI ON DEMAND PIPELINE
         ===========================
         Model          : ${params.model}
         Task           : ${params.task}
         Model config   : ${params.model_config}
         Image filepaths: ${params.img_dir}
         Executor       : ${params.executor}
         """.stripIndent()

workflow {
    img_ch = Channel.fromPath( params.img_dir ).splitText()


    // Select appropriate model
    if( params.model == "sam" )
        RUNSAM(img_ch, params.model_config)

    else
        error "Model ${params.model} not yet implemented!"
}

workflow.onComplete{
    log.info ( workflow.success ? '\nDone!' : '\nSomething went wrong!' )
}