# napri-nxf

[![License BSD-3](https://img.shields.io/pypi/l/napri-nxf.svg?color=green)](https://github.com/MatousE/napri-nxf/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napri-nxf.svg?color=green)](https://pypi.org/project/napri-nxf)
[![Python Version](https://img.shields.io/pypi/pyversions/napri-nxf.svg?color=green)](https://python.org)
[![tests](https://github.com/MatousE/napri-nxf/workflows/tests/badge.svg)](https://github.com/MatousE/napri-nxf/actions)
[![codecov](https://codecov.io/gh/MatousE/napri-nxf/branch/main/graph/badge.svg)](https://codecov.io/gh/MatousE/napri-nxf)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napri-nxf)](https://napari-hub.org/plugins/napri-nxf)

A plugin that runs a nextflow pipeline from a napari plugin

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation
Create conda environment:
    
    conda create -n nxf_env python=3.9
    
Then ensure that this environment is active for the following steps.

### Napari on Windows/Linux/Mac (Intel)

    pip install "napari[all]"

### Napari on Mac (Apple Silicon, M1/M2 etc.)
`napari` uses PyQt5 by default, but you'll need to install it using `conda` (or `mamba` if `conda` is being slow)

    conda install pyqt
    pip install napari

If issues still occur, then use PyQt6 (which has wheels for Apple silicon):

    pip install napari pyqt6

### Nextflow
Install nextflow

    conda install -c bioconda nextflow
    
Install Python nextflow wrapper

    pip install nextflowpy

### This Plugin
Clone this plugin

    git clone https://github.com/FrancisCrickInstitute/ai-on-demand.git

Move to `napari-ai-od/` directory

    cd napari-ai-od

Install plugin

    pip install -e .

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napri-nxf" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
