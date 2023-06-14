
process RUNSAM {
    input:
    path image_path
    path model_config
    
    output:
    stdout

    script:
    """
    run_sam.py --path ${image_path} --config ${model_config}
    """
}