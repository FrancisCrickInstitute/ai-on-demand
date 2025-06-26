# AIoD Napari Plugin

A Napari plugin, part of [AI OnDemand](https://github.com/FrancisCrickInstitute/AIoD-Model-Registry/wiki), to provide an accessible interface for running deep learning models on images via our [Nextflow pipeline](https://github.com/FrancisCrickInstitute/Segment-Flow).

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation
### Pip
You can install this plugin using `pip`, either from the repo directly:

    pip install git+https://github.com/FrancisCrickInstitute/ai-on-demand.git

or clone and install locally:

    git clone https://github.com/FrancisCrickInstitute/ai-on-demand.git
    cd ai-on-demand
    pip install .

### Conda
We have provided a conda environment file to install all the dependencies for this plugin. To install the environment, run the following command:

    conda env create -f ai-od.yml

Note that when it comes to the installation of napari this may be preferable, depending on whether your system is best supported by the pip- or conda-packaged version.

### Napari
If you have any issues installing Napari, see their [installation guide](https://napari.org/stable/tutorials/fundamentals/installation.html). Then, you can reinstall this package following the instructions above.

### Nextflow
This plugin makes a call to Nextflow when running the pipeline. To install Nextflow, see their [installation guide](https://www.nextflow.io/docs/latest/install.html).

If you have any issues with that, there is a [version of Nextflow in bioconda](https://anaconda.org/bioconda/nextflow) that you can use instead, but it may not be the most up-to-date version.

## Usage
For general usage of the plugin, see the [user guide here](user_guide.md).

For developers, see our [developer guide](developer_guide.md) for some tips on how to get started and contribute to the plugin.

For more information on the AI OnDemand project, see our [wiki](https://github.com/FrancisCrickInstitute/AIoD-Model-Registry/wiki).

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure the coverage at least stays the same before you submit a pull request.

## License

Currently GPL-3.0 licensed. See `LICENSE` for more information. Pending further discussion.

## Issues

If you encounter any problems, please raise an issue along with a detailed description.