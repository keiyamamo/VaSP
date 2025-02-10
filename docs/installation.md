# Installation

## Dependencies

The dependencies of `VaSP` are:

* Python >= 3.10
* cppimport >= 22.8.2
* pandas >= 2.1.2
* matplotlib >= 3.8.0
* FEniCS >= 2018.1
* morphMan >= 1.2
* VaMPy >= 1.0.4
* vmtk >= 1.4.0
* turtleFSI >= 2.4.0

## Installing with `conda`

To install `VaSP` and all its dependencies to a *local environment*, we recommend using `conda`.
Instructions for installing `VaSP`
with `conda` can be found [here](install:conda).

## Installing with Docker

To install `VaSP` and all its dependencies to an *isolated environment*, we recommend using the dedicated Docker
container. Instructions for installing `VaSP` with Docker can be found [here](install:docker).

## Installing in high-performance computing (HPC) clusters

To install and use `VaSP`, it is recommended to first install `FEniCS` on the cluster and then install `turtleFSI`, `VaMPy`, and `VaSP` separately. To install `FEniCS`, it is recommended to build it from source and instructions can be found [here](https://fenics.readthedocs.io/en/latest/installation.html). After installing `FEniCS`, you can install `turtleFSI`, `VaMPy`, and `VaSP` via `pip`.