# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import argparse
from dolfin import MPI, VectorFunctionSpace, Function, Mesh, HDF5File, parameters

from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names

from fsipy.simulations.simulation_common import 

# set compiler arguments
parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 6


def parse_arguments():
    """Read arguments from commandline"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--folder", type=Path, help="Path to simulation results folder")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file. If not given (None), " +
                             "it will assume that mesh is located <folder>/Mesh/mesh.h5)")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of output data")
    args = parser.parse_args()

    return args


def compute_stress(visualization_separate_domain_folder: Path, mesh_path: Path, stride: int) -> None:
    """
  

    Args:
        visualization_separate_domain_folder (Path): Path to the folder containing d.h5 (or d_solid.h5) file
        mesh_path (Path): Path to the mesh file (non-refined, whole domain)
        stride (int): Save frequency of output data
        solid_properties (list): List of dictionaries containing solid properties used in the simulation
        fluid_properties (list): List of dictionaries containing fluid properties used in the simulation
    """
    # find the displacement file and check if it is for the entire domain or only for the solid domain
    try:
        file_path_d = visualization_separate_domain_folder / "d_solid.h5"
        assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
        solid_only = True
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using d_solid.h5 file \n")
    except AssertionError:
        file_path_d = visualization_separate_domain_folder / "d.h5"
        assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
        solid_only = False
        if MPI.rank(MPI.comm_world) == 0:
            print("--- displacement is for the entire domain \n")

    file_d = HDF5File(MPI.comm_world, str(file_path_d), "r")

    with HDF5File(MPI.comm_world, str(file_path_d), "r") as f:
        dataset = get_dataset_names(f, step=stride, vector_filename="/displacement/vector_%d")

    # Read the original mesh and also the refined mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Read the refined mesh \n")

    mesh_name = mesh_path.stem
    refined_mesh_path = mesh_path.parent / f"{mesh_name}_refined_solid.h5" if solid_only else \
        mesh_path.parent / f"{mesh_name}_refined.h5"
    refined_mesh = Mesh()
    with HDF5File(MPI.comm_world, str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(refined_mesh, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Create function space for the displacement on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    d_p1 = Function(Vv_refined)
  
    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    counter = 0
    for data in dataset:
        # Read diplacement data and interpolate to P2 space
        file_d.read(d_p1, data)
        

        t = file_d.attributes(dataset[counter])["timestamp"]
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, f"Calculating Stress & Strain at Timestep: {t}", "=" * 10)
        



def main() -> None:
    """Main function."""
    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    args = parse_arguments()
    folder_path = args.folder

    visualization_separate_domain_folder = args.folder / "Visualization_separate_domain"
    assert (
        visualization_separate_domain_folder.exists()
    ), f"Visualization_separate_domain folder {visualization_separate_domain_folder} not found."

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using mesh from default turrtleFSI Mesh folder \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    compute_stress(visualization_separate_domain_folder, mesh_path, args.stride)


if __name__ == "__main__":
    main()
