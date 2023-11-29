# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#  Kei Yamamoto

"""
This script computes hemodynamic indices from the velocity field.
It is assumed that the user has already run create_hdf5.py to create the hdf5 files 
and obtained u.h5 in the Visualization_separate_domain folder.
"""

from pathlib import Path
import argparse

from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters, XDMFFile, TrialFunction, TestFunction, \
    inner, ds, assemble, FacetNormal, sym, project, FunctionSpace, VectorElement, PETScDMCollection, grad, sym, solve
from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names
from fsipy.automatedPostprocessing.postprocessing_common import read_parameters_from_file
from fsipy.automatedPostprocessing.postprocessing_fenics.postprocessing_fenics_common import project_dg

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["reorder_dofs_serial"] = False


def parse_arguments():
    """Read arguments from commandline"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--folder', type=Path, help="Path to simulation results folder")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file. If not given (None), it will assume that mesh is located <folder_path>/Mesh/mesh.h5)")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of output data")
    args = parser.parse_args()

    return args


#TODO: optimize the function
def _surface_project(f, V):
    """
    Project a function contains surface integral onto a function space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    a_proj = inner(u, v)*ds
    b_proj = inner(f, v)*ds
    # keep_diagonal=True & ident_zeros() are necessary for the matrix to be invertible
    A = assemble(a_proj, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(b_proj)
    u = Function(V)
    solve(A, u.vector(), b)
    return u


class Stress:
    def __init__(self, u, mu, mesh, velocity_degree):
        self.V = VectorFunctionSpace(mesh, 'DG', velocity_degree -1)

        # Compute stress tensor
        sigma = (2 * mu * sym(grad(u)))

        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        self.Ft = F - (Fn * n)  # vector-valued

    def __call__(self):
        """
        Compute stress for given velocity field u

        Returns:
            Ftv_mb (Function): Shear stress
        """
        self.Ftv = _surface_project(self.Ft, self.V)

        return self.Ftv
    


def compute_hemodyanamics(visualization_separate_domain_path, mesh_path, mu, stride=1):

    """
    Args:
        mesh_name: Name of the input mesh for the simulation. This function will find the fluid only mesh based on this name
        case_path (Path): Path to results from simulation
        nu (float): Viscosity
        dt (float): Actual ime step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)
    """
    

    file_path_u = visualization_separate_domain_folder / "u.h5"
    assert file_path_u.exists(), f"Velocity file {file_path_u} not found.  Make sure to run create_hdf5.py first."
    file_u = HDF5File(MPI.comm_world, str(file_path_u), "r")

    with HDF5File(MPI.comm_world, str(file_path_u), "r") as f:
        dataset = get_dataset_names(f, step=stride, vector_filename="/velocity/vector_%d")
    
    # Read the original mesh and also the refined mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Read the original mesh and also the refined mesh \n")

    fluid_mesh_path = mesh_path / "mesh_fluid.h5"
    mesh = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_mesh_path), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    refined_mesh_path = mesh_path / "mesh_refined_fluid.h5"
    refined_mesh = Mesh()
    with HDF5File(MPI.comm_world,  str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(refined_mesh, "mesh", False)

   # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Create function space for the velocity on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    # Create function space for the velocity on the refined mesh with P2 elements
    Vv_non_refined = VectorFunctionSpace(mesh, "CG", 2)

    # Create function space for hemodynamic indices with DG1 elements
    Vv = VectorFunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "DG", 1)

    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define functions")

    # Create functions

    # u_p2 is the velocity on the refined mesh with P2 elements
    u_p2 = Function(Vv_non_refined)
    # u_p1 is the velocity on the refined mesh with P1 elements
    u_p1 = Function(Vv_refined)

    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    u_transfer_matrix = PETScDMCollection.create_transfer_matrix(Vv_refined, Vv_non_refined)

    # Time-dependent wall shear stress
    WSS = Function(Vv)
    
    # Relative residence time 
    RRT = Function(V)

    # Oscillatory shear index
    OSI = Function(V)

    # Endothelial cell activation potential
    ECAP = Function(V)

    # Time averaged wall shear stress and mean WSS magnitude
    TAWSS = Function(V)
    WSS_mean = Function(Vv)

    # Temporal wall shear stress gradient
    TWSSG = Function(V)
    twssg = Function(Vv)
    tau_prev = Function(Vv)

    # Define stress object with P2 elements and non-refined mesh
    stress = Stress(u_p2, mu, mesh, 2)

    # Create XDMF files for saving indices
    hemodynamic_indices_path = visualization_separate_domain_path.parent / "Hemodynamic_indices"
    hemodynamic_indices_path.mkdir(parents=True, exist_ok=True)
    index_names = ["RRT", "OSI", "ECAP", "WSS", "TAWSS", "TWSSG"]
    index_variables = [RRT, OSI, ECAP, WSS, TAWSS, TWSSG]
    index_dict = dict(zip(index_names, index_variables))
    xdmf_paths = [hemodynamic_indices_path / f"{name}.xdmf" for name in index_names]

    indices = {}
    for index, path in zip(index_names, xdmf_paths):
        indices[index] = XDMFFile(MPI.comm_world, str(path))
        indices[index].parameters["rewrite_function_mesh"] = False
        indices[index].parameters["flush_output"] = True
        indices[index].parameters["functions_share_mesh"] = True

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Get time difference between two consecutive time steps
    dt = file_u.attributes(dataset[1])["timestamp"] - file_u.attributes(dataset[0])["timestamp"]

    counter = 0
    for data in dataset:
        # Read velocity data and interpolate to P2 space
        file_u.read(u_p1, data)
        u_p2.vector()[:] = u_transfer_matrix * u_p1.vector()

        t = file_u.attributes(dataset[counter])["timestamp"]
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, f"Calculating WSS at Timestep: {t}", "=" * 10)

        # compute WSS and accumulate for time-averaged WSS
        tau = stress()

        # Write temporal WSS
        tau.rename("WSS", "WSS")
        indices["WSS"].write_checkpoint(tau, "WSS", t, XDMFFile.Encoding.HDF5, append=True)

        # Compute time-averaged WSS by accumulating WSS magnitude
        tawss = project(inner(tau, tau) ** (1 / 2), V)
        TAWSS.vector().axpy(1, tawss.vector())

        # Simply accumulate WSS for computing OSI and ECAP later
        WSS_mean.vector().axpy(1, tau.vector())
    
        # Compute TWSSG
        twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt)
        twssg.vector().apply("insert")
        twssg_ = project_dg(inner(twssg, twssg) ** (1 / 2) , V)
        TWSSG.vector().axpy(1, twssg_.vector())
    
        # Update tau    
        tau_prev.vector().zero()
        tau_prev.vector().axpy(1, tau.vector())

        counter += 1
    
    indices["WSS"].close()

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Saving hemodynamic indices", "=" * 10)

    index_dict['TWSSG'].vector()[:] = index_dict['TWSSG'].vector()[:] / counter
    index_dict['TAWSS'].vector()[:] = index_dict['TAWSS'].vector()[:] / counter
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / counter
    wss_mean = project(inner(WSS_mean, WSS_mean) ** (1 / 2), V)
    wss_mean_vec = wss_mean.vector().get_local()
    tawss_vec = index_dict['TAWSS'].vector().get_local()
    from IPython import embed; embed(); exit(1)
    # Compute RRT, OSI, and ECAP based on mean and absolute WSS
    index_dict['RRT'].vector().set_local(1 / wss_mean_vec)
    index_dict['OSI'].vector().set_local(0.5 * (1 - wss_mean_vec / tawss_vec))
    index_dict['ECAP'].vector().set_local(index_dict['OSI'].vector().get_local() / tawss_vec)

    for index in ['RRT', 'OSI', 'ECAP']:
        index_dict[index].vector().apply("insert")

    # Rename displayed variable names
    for name, var in index_dict.items():
        var.rename(name, name)

    # Write indices to file
    for name, xdmf_object in indices.items():
        index = index_dict[name]
        if name == "WSS":
            pass
        else:
            indices[name].write_checkpoint(index, name, 0, XDMFFile.Encoding.HDF5, append=False)
            indices[name].close()
            if MPI.rank(MPI.comm_world) == 0:
                print(f"--- {name} is saved in {hemodynamic_indices_path}")

    file_u.close()

        


if __name__ == '__main__':

    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    args = parse_arguments()
    # Define paths for visulization and mesh files
    folder_path = Path(args.folder)
    assert folder_path.exists(), f"Folder {folder_path} not found."
    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"
    assert visualization_separate_domain_folder.exists(), f"Folder {visualization_separate_domain_folder} not found. " \
                                                            "Please make sure to run create_hdf5.py first."

    parameters = read_parameters_from_file(args.folder)
    save_deg = parameters["save_deg"]
    assert save_deg == 2, "This script only works for save_deg = 2"

    mu_f = parameters["mu_f"]

    if isinstance(mu_f, list):
        if MPI.rank(MPI.comm_world) == 0:
            print("--- two fluid regions are detected. Using the first fluid region for viscosity \n")
        mu = mu_f[0]

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using mesh from default turrtleFSI Mesh folder \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    compute_hemodyanamics(visualization_separate_domain_folder, mesh_path, mu, args.stride)
