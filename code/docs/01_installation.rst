.. _installation-label:

Installation
==============================

This page describes the complete installation process for the ML4PdM library. You will be guided through all the steps that are necessary to be able to use the library and run the examples.

Cloning Sources
------------------------------

Currently there is no package for this library which means that you have to install the sources first. Use one of the following commands to clone the Git repository:

.. code-block:: bash

    # SSH:
    git clone irb-git@git.cs.uni-paderborn.de:machine-learning-for-predictive-maintenance/code.git
    # HTTPS:
    git clone https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/code.git

Installing Anaconda
------------------------------

To install all dependencies, a recent version of Anaconda needs to be installed. It is also possible to install the dependencies via other tools, but this is not tested and therefore not recommended.

First, download the `Anaconda Individual Edition <https://www.anaconda.com/products/individual>`_ for your current operating system. Next, follow the `official documentation <https://docs.anaconda.com/anaconda/install/>`_ to install Anaconda on your system. After you have installed Anaconda successfully and verified the installation like described in the official documentation, you might want to add Anaconda to your PATH for easier access outside of the Anaconda Prompt. **Note:** There might be some side effects to that which are described in the `Anaconda FAQ <https://docs.anaconda.com/anaconda/user-guide/faq/#installing-anaconda>`_.

Installing Dependencies
------------------------------

There are three levels of dependencies that can be installed depending on what you are trying to achieve. The dependencies are organized in three YML files. This allows you to only install a limited number of dependencies which perfectly fit your needs.

* **ml4pdm.yml:** Contains the main dependencies that are absolutely necessary to use the library to its full extent.
* **examples.yml:** Contains additional dependencies that are optional and only needed to execute the examples that are presented in the ``examples/`` folder.
* **development.yml:** Contains additional dependencies that are optional and only needed when you are planning to actively develop the library.

Please decide which of the above dependencies you want to install and make sure that you are in the root directory of this project. Execute the following commands in the given order because for example the tensorflow version will be using NVIDIAs CUDA when you install the ``examples.yml`` first. You can skip the lines/files that are not necessary for your work.

**Note:** For users of the operating system macOS there are separate files for ``ml4pdm.yml`` and ``examples.yml``. They can be found in ``.conda/macOS/``. Please adapt the paths below accordingly.

.. code-block:: bash

    conda create -n ml4pdm
    # Optional:
    conda env update -n ml4pdm --file ".conda/examples.yml" --prune
    conda env update -n ml4pdm --file ".conda/development.yml" --prune
    # Mandatory:
    conda env update -n ml4pdm --file ".conda/ml4pdm.yml" --prune
    conda activate ml4pdm

Verify installation
------------------------------

To check whether the installation was successful, you can try to execute the code from our :ref:`examples-label` page. You can also test more examples that are found in the ``examples/`` folder if you installed the corresponding dependencies above. If you are using a Jupyter server for running the examples, you need to run the commands below (more details `here <https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084>`_).

.. code-block:: bash

    conda activate ml4pdm
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=ml4pdm

For macOS users the hyperparameter optimization cells have to be skipped as ``ray-tune`` is currently not available for macOS in the conda-forge repository.
