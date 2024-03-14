# MHD-IMEXRK-paper
This repository contains the source code utilized to analyse implicit-explicit time integration schemes for a 2d magneto-hydrodynamic solver relevant to the research paper:

- A. Piterskaya, M. Mortensen; "A comparison of implicit-explicit Runge-Kutta time integration schemes in numerical solvers developed based on the Galerkin and Petrov-Galerkin spectral methods for two-dimensional magneto-hydrodynamic problems", 2024

The model described in the paper has been implemented within the spectral Galerkin framework Shenfun (https://github.com/spectralDNS/shenfun), version 4.1.1.

To facilitate the conda installation process, kindly refer to the 'environment.yml' file, which contains a comprehensive list of dependencies required to establish a fully operational shenfun environment.

# Codespace

The code in this repository can be tested using a codespace. Press the green Code button and choose to "create a codespace on main". A virtual machine will then be created with all the required software in environment.yml installed in a coda environment. To activate this do

    source activate ./venv

in the terminal of the codespace after the installation is finished. You may then run the program using

    python BurgersForward.py

