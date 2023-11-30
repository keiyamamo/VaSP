# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#  Kei Yamamoto 2023

"""
This script computes hemodynamic indices from the velocity field.
It is assumed that the user has already run create_hdf5.py to create the hdf5 files
and obtained u.h5 in the Visualization_separate_domain folder.
"""
import numpy as np
from pathlib import Path
import argparse

from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters, XDMFFile, TrialFunction, \
    TestFunction, inner, ds, assemble, FacetNormal, sym, project, FunctionSpace, PETScDMCollection, grad, solve, \
    BoundaryMesh
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
                        help="Path to the mesh file. If not given (None)," +
                             "it will assume that mesh is located <folder_path>/Mesh/mesh.h5)")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of output data")
    args = parser.parse_args()

    return args


def _surface_project(f, V):
    """
    Project a function contains surface integral onto a function space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    a_proj = inner(u, v) * ds
    b_proj = inner(f, v) * ds
    # keep_diagonal=True & ident_zeros() are necessary for the matrix to be invertible
    A = assemble(a_proj, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(b_proj)
    u_ = Function(V)
    solve(A, u_.vector(), b)
    return u_


def _interpolate_dg(sub_map, sub_dofmap, V1, V_sub, mesh, v_sub_copy, u_vec, sub_coords, dof_coords):

    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim())
    f_to_c = mesh.topology()(mesh.topology().dim() - 1, mesh.topology().dim())
    for i, facet in enumerate(sub_map):
        cells = f_to_c(facet)
        # Get closure dofs on parent facet

        sub_dofs = sub_dofmap.cell_dofs(i)
        closure_dofs = V1.dofmap().entity_closure_dofs(
            mesh, mesh.topology().dim(), [cells[0]])
        copy_dofs = np.empty(len(sub_dofs), dtype=np.int32)

        for dof in closure_dofs:
            for j, sub_coord in enumerate(sub_coords[sub_dofs]):
                if np.allclose(dof_coords[dof], sub_coord):
                    copy_dofs[j] = dof
                    break
        sub_dofs = sub_dofmap.cell_dofs(i)
        # Copy data
        v_sub_copy[sub_dofs] = u_vec[copy_dofs]

    return v_sub_copy


class Stress:
    def __init__(self, u, mu, mesh, boundary_mesh, velocity_degree):
        self.V = VectorFunctionSpace(mesh, 'DG', velocity_degree - 1)
        self.Vb = VectorFunctionSpace(boundary_mesh, 'DG', velocity_degree - 1)
        self.Ftv_b = Function(self.Vb)
        self.sub_map = boundary_mesh.entity_map(mesh.topology().dim() - 1).array()
        self.sub_dofmap = self.Vb.dofmap()
        self.mesh = mesh
        v_sub = Function(self.Vb)
        self.v_sub_copy = v_sub.vector().get_local()
        self.sub_coords = self.Vb.tabulate_dof_coordinates()
        self.dof_coords = self.V.tabulate_dof_coordinates()

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

        self.v_sub_copy = _interpolate_dg(self.sub_map, self.sub_dofmap, self.V, self.Vb, self.mesh,
                                          self.v_sub_copy, self.Ftv.vector().get_local(), self.sub_coords,
                                          self.dof_coords)
        self.Ftv_b.vector().set_local(self.v_sub_copy)

        return self.Ftv_b


def compute_hemodyanamics(visualization_separate_domain_path, mesh_path, mu, stride=1):

    """
    Args:
        visualization_separate_domain_path (Path): Path to the folder containing u.h5
        mesh_path (Path): Path to the mesh folder
        mu (float): Dynamic viscosity
        stride (int): reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
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
    with HDF5File(MPI.comm_world, str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(refined_mesh, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Create function space for the velocity on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    # Create function space for the velocity on the refined mesh with P2 elements
    Vv_non_refined = VectorFunctionSpace(mesh, "CG", 2)

    # Create boundary mesh and function space
    boundary_mesh = BoundaryMesh(mesh, "exterior")

    # Create function space for hemodynamic indices with DG1 elements
    Vv_boundary_mesh = VectorFunctionSpace(boundary_mesh, "DG", 1)
    V_boundary_mesh = FunctionSpace(boundary_mesh, "DG", 1)

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
    WSS = Function(Vv_boundary_mesh)

    # Relative residence time
    RRT = Function(V_boundary_mesh)

    # Oscillatory shear index
    OSI = Function(V_boundary_mesh)

    # Endothelial cell activation potential
    ECAP = Function(V_boundary_mesh)

    # Time averaged wall shear stress and mean WSS magnitude
    TAWSS = Function(V_boundary_mesh)
    WSS_mean = Function(Vv_boundary_mesh)

    # Temporal wall shear stress gradient
    TWSSG = Function(V_boundary_mesh)
    twssg = Function(Vv_boundary_mesh)
    tau_prev = Function(Vv_boundary_mesh)

    # Define stress object with P2 elements and non-refined mesh
    stress = Stress(u_p2, mu, mesh, boundary_mesh, 2)

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
        tawss = project(inner(tau, tau) ** (1 / 2), V_boundary_mesh)
        TAWSS.vector().axpy(1, tawss.vector())

        # Simply accumulate WSS for computing OSI and ECAP later
        WSS_mean.vector().axpy(1, tau.vector())

        # Compute TWSSG
        twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt)
        twssg.vector().apply("insert")
        twssg_ = project_dg(inner(twssg, twssg) ** (1 / 2), V_boundary_mesh)
        TWSSG.vector().axpy(1, twssg_.vector())

        # Update tau
        tau_prev.vector().zero()
        tau_prev.vector().axpy(1, tau.vector())

        counter += 1

    indices["WSS"].close()
    file_u.close()

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Saving hemodynamic indices", "=" * 10)

    index_dict['TWSSG'].vector()[:] = index_dict['TWSSG'].vector()[:] / counter
    index_dict['TAWSS'].vector()[:] = index_dict['TAWSS'].vector()[:] / counter
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / counter
    wss_mean = project(inner(WSS_mean, WSS_mean) ** (1 / 2), V_boundary_mesh)
    wss_mean_vec = wss_mean.vector().get_local()
    tawss_vec = index_dict['TAWSS'].vector().get_local()

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
