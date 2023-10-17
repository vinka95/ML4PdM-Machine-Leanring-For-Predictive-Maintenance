# Setup Anaconda Environment

This repository contains conda environment file which have the all library list that is required by ML4PdM projects.

# How to setup Anaconda Environment from the .yml files?
* Download the [Anaconda Individual Edition](https://www.anaconda.com/products/individual).
* Check Operating System and 32 or 64 bit version software for your PC. It will take some time to download the installer file.
* Double click on .exe file and follow the prompt window and don’t change the default settings. (https://www.youtube.com/watch?v=C4OPn58BLaU)
* Run the following commands in the root directory of this repository:
  ```
  conda env create -f .conda/examples.yml
  conda env update -n ml4pdm --file .conda/development.yml --prune
  conda env update -n ml4pdm --file .conda/ml4pdm.yml --prune
  ```

# How to update the Anaconda Environment?
* Run the following commands in the root directory of this repository:
  ```
  conda env update -n ml4pdm --file .conda/ml4pdm.yml --prune
  conda env update -n ml4pdm --file .conda/examples.yml --prune
  conda env update -n ml4pdm --file .conda/development.yml --prune
  ```

# How to install new packages?
* First run `conda install -c conda-forge <package-name>` and **do not** confirm (y) the installation yet
* After the environment was solved, check the exact version that will be installed and add it to the corresponding `.yml` file
* Watch out for packages that are downgraded or upgraded an update the `.yml` files accordingly
* Now you can confirm the installation and let the process finish