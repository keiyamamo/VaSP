# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script loads displacement and velocity from .h5 file given by create_hdf5.py,
converts and saves to .xdmf format for visualization (in e.g. ParaView).
"""

from pathlib import Path
import argparse

from dolfin import Mesh, HDF5File, FunctionSpace, Function, MPI, parameters, XDMFFile
from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names


# set compiler arguments
parameters["reorder_dofs_serial"] = False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder", type=Path, help="Path to simulation results")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file. If not given (None), " +
                             "it will assume that mesh is located <folder>/Mesh/mesh.h5)")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of simulation")
    parser.add_argument("--high-pass", action="store_true", help="data is high-pass filtered")
    parser.add_argument("--bands", type=int, nargs="+", default=[1, 2], help="band numbers for high-pass filtering")
    return parser.parse_args()


def create_separate_domain_visualization(visualization_separate_domain_folder, mesh_path, stride=1, 
                                         high_pass=False, bands=[1, 2]):
    """
    Loads displacement and velocity from .h5 file given by create_hdf5.py,
    converts and saves to .xdmf format for visualization (in e.g. ParaView).
    This function works with MPI. If the displacement was saved for the entire domain,
    no additional xdmf file will be created for the displacement.
    Args:
        visualization_separate_domain_folder (Path): Path to the folder containing the .h5 files
        mesh_path (Path): Path to the mesh that contains both fluid and solid domain
        stride (int): Save frequency of visualization output (default: 1)
    """
    # Path to the input files
    if high_pass:
        file_path_p = visualization_separate_domain_folder / f"p_{bands[0]}_to_{bands[1]}.h5"
    else:
        file_path_p = visualization_separate_domain_folder / "p.h5"
    assert file_path_p.exists() 
   
    file_p = HDF5File(MPI.comm_world, str(file_path_p), "r")
    dataset_p = get_dataset_names(file_p, step=stride, vector_filename="/pressure/vector_%d")

    # Define mesh path related variables
    fluid_domain_path = mesh_path.with_name(mesh_path.stem + "_fluid.h5")
    assert fluid_domain_path.exists(), f"Fluid mesh file {fluid_domain_path} not found."

    # Read fluid and solid mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Reading fluid mesh file \n")

    mesh_fluid = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_domain_path), "r") as mesh_file:
        mesh_file.read(mesh_fluid, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Vf = VectorFunctionSpace(mesh_fluid, "CG", 1)
    Vf = FunctionSpace(mesh_fluid, "CG", 1)
    p = Function(Vf)
    
    # Create writer for displacement and velocity
    if high_pass:
        p_path = visualization_separate_domain_folder / f"pressure_{bands[0]}_to_{bands[1]}.xdmf"
    else:
        p_path = visualization_separate_domain_folder / "pressure_fluid.xdmf"
    p_writer = XDMFFile(MPI.comm_world, str(p_path))
    p_writer.parameters["flush_output"] = True
    p_writer.parameters["functions_share_mesh"] = False
    p_writer.parameters["rewrite_function_mesh"] = False


    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    for i in range(len(dataset_p)):
        
        file_p.read(p, dataset_p[i])

        timestamp = file_p.attributes(dataset_p[i])["timestamp"]
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, "Timestep: {}".format(timestamp), "=" * 10)

        p.rename("pressure", "pressure")
        p_writer.write(p, timestamp)

    p_writer.close()
 
    if MPI.rank(MPI.comm_world) == 0:
        print("========== Post processing finished ========== \n")
        print(f"--- Visualization files are saved in: {visualization_separate_domain_folder.absolute()}")


def main() -> None:
    args = parse_arguments()

    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    # Define paths for visulization and mesh files
    folder_path = Path(args.folder)
    assert folder_path.exists(), f"Folder {folder_path} not found."
    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using non-refined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    create_separate_domain_visualization(visualization_separate_domain_folder, mesh_path, args.stride, 
                                         args.high_pass, args.bands)


if __name__ == "__main__":
    main()
