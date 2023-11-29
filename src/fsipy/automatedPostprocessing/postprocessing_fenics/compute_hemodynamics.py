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
    inner, ds, assemble, FacetNormal, sym, project, FunctionSpace, VectorElement, PETScDMCollection
from fsipy.automatedPostprocessing.postprocessing_common import read_parameters_from_file

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["reorder_dofs_serial"] = False


def parse_arguments():
    """Read arguments from commandline"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--folder', type=Path, help="Path to simulation results folder")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file (default: <folder_path>/Mesh/mesh.h5)")
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
    def __init__(self, u, nu, mesh, velocity_degree):
        self.V = VectorFunctionSpace(mesh, 'DG', velocity_degree -1)

        # Compute stress tensor
        sigma = (2 * nu * sym(grad(u)))

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
    


def compute_hemodyanamics(visualization_separate_domain_path, mesh_path, nu):

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


    with HDF5File(MPI.comm_world, str(file_path_u), "r") as f:
        dataset_u = get_dataset_names(f, step=stride, vector_filename="/velocity/vector_%d")
    
    # Read the original mesh and also the refined mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Read the original mesh and also the refined mesh \n")

    mesh = Mesh()
    with HDF5File(MPI.comm_world, str(mesh_path), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    refined_mesh_path = mesh_path.with_name(mesh_path.stem + "_refined.h5")
    refined_mesh = Mesh()
    with HDF5File(MPI.comm_world,  str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(mesh_viz, "mesh", False)

   # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Create function space for the velocity on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    # Create function space for the velocity on the refined mesh with P2 elements
    Vv = VectorFunctionSpace(mesh, "CG", 2)

    # Create function space for hemodynamic indices with DG1 elements
    Vv = VectorFunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "DG", 1)

    if MPI.rank(MPI.comm_world) == 0:
        print("Define functions")

    # Create functions

    # u_p2 is the velocity on the refined mesh with P2 elements
    u_p2 = Function(Vv)
    # u_p1 is the velocity on the refined mesh with P1 elements
    u_p1 = Function(Vv_refined)

    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    u_transfer_matrix = PETScDMCollection.create_transfer_matrix(Vv_refined, Vv)
    
    # Relative residence time 
    RRT = Function(V)
    RRT_avg = Function(V)

    # Oscillatory shear index
    OSI = Function(V)
    OSI_avg = Function(V)

    # Endothelial cell activation potential
    ECAP = Function(V)
    ECAP_avg = Function(V)

    # WSS_mean
    WSS_mean = Function(Vv)
    WSS_mean_avg = Function(Vv)

    # Time averaged wall shear stress
    TAWSS = Function(V)
    TAWSS_avg = Function(V)

    # Temporal wall shear stress gradient
    TWSSG = Function(U_b1)
    TWSSG_avg = Function(U_b1)
    twssg = Function(V_b1)
    tau_prev = Function(V_b1)

    # Define stress object with P2 elements and non-refined mesh
    stress = Stress(u_p2, nu, mesh, 2)

    WSS_file = XDMFFile(MPI.comm_world, WSS_ts_path)

    WSS_file.parameters["flush_output"] = True
    WSS_file.parameters["functions_share_mesh"] = True
    WSS_file.parameters["rewrite_function_mesh"] = False

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # NOTE: Inside the loop, there is a lof ot projection between different function spaces. This is not efficient.    
    while True:
        # Read in velocity solution to vector function u
        try:
            f = HDF5File(MPI.comm_world, file_path_u.__str__(), "r")
            vec_name = "/velocity/vector_%d" % file_counter
            t = f.attributes(vec_name)["timestamp"]


            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Calculating WSS at Timestep: {}".format(t), "=" * 10)
            f.read(u_viz, vec_name)

            # Calculate v in P2 based on visualization refined P1
            u.vector()[:] = dv_trans*u_viz.vector()
        
            # Compute WSS
            if MPI.rank(MPI.comm_world) == 0:
                print("Compute WSS (mean)")
            tau = stress()     
        
            # tau.vector()[:] = tau.vector()[:] * 1000 # Removed this line, presumably a unit conversion
            WSS_mean.vector().axpy(1, tau.vector())
        
            if MPI.rank(MPI.comm_world) == 0:
                print("Compute WSS (absolute value)")
        
            wss_abs = project(inner(tau,tau)**(1/2),U_b1) # Calculate magnitude of Tau (wss_abs)
            WSS_abs.vector().axpy(1, wss_abs.vector())  # WSS_abs (cumulative, added together)
            # axpy : Add multiple of given matrix (AXPY operation)
        
            # Name functions
            wss_abs.rename("Wall Shear Stress", "WSS_abs")
        
            # Write results
            WSS_file.write_checkpoint(wss_abs, "wss_abs", t, XDMFFile.Encoding.HDF5, True)
        
            # Compute TWSSG
            if MPI.rank(MPI.comm_world) == 0:
                print("Compute TWSSG")
            twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt) # CHECK if this needs to be the time between files or the timestep of the simulation...
            twssg.vector().apply("insert")
            twssg_ = project(inner(twssg,twssg)**(1/2),U_b1) # Calculate magnitude of TWSSG vector
            TWSSG.vector().axpy(1, twssg_.vector())
        
            # Update tau
            if MPI.rank(MPI.comm_world) == 0:
                print("Update WSS \n")
            tau_prev.vector().zero()
            tau_prev.vector().axpy(1, tau.vector())
            n += 1
                
            # Update file_counter
            file_counter += stride

        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero

            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break   

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Saving hemodynamic indices", "=" * 10)

    TWSSG.vector()[:] = TWSSG.vector()[:] / n
    WSS_abs.vector()[:] = WSS_abs.vector()[:] / n
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / n

    WSS_abs.rename("WSS", "WSS")
    TWSSG.rename("TWSSG", "TWSSG")

    try:
        wss_mean = project(inner(WSS_mean,WSS_mean)**(1/2),U_b1) # Calculate magnitude of WSS_mean vector
        wss_mean_vec = wss_mean.vector().get_local()
        wss_abs_vec = WSS_abs.vector().get_local()

        # Compute RRT and OSI based on mean and absolute WSS
        RRT.vector().set_local(1 / wss_mean_vec)
        RRT.vector().apply("insert")
        RRT.rename("RRT", "RRT")

        OSI.vector().set_local(0.5 * (1 - wss_mean_vec / wss_abs_vec))
        OSI.vector().apply("insert")
        OSI.rename("OSI", "OSI")
        save = True
    except:
        if MPI.rank(MPI.comm_world) == 0:
            print("Failed to compute OSI and RRT")
        save = False

    if save:
        # Save OSI and RRT
        rrt_path = (visualization_separate_domain_path / "RRT.xdmf").__str__()
        osi_path = (visualization_separate_domain_path / "OSI.xdmf").__str__()

        rrt = XDMFFile(MPI.comm_world, rrt_path)
        osi = XDMFFile(MPI.comm_world, osi_path)

        for f in [rrt, osi]:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            f.parameters["rewrite_function_mesh"] = False

        rrt.write_checkpoint(RRT, "RRT", 0)
        osi.write_checkpoint(OSI, "OSI", 0)

    # Save WSS and TWSSG
    wss_path = (visualization_separate_domain_path / "WSS.xdmf").__str__()
    twssg_path = (visualization_separate_domain_path / "TWSSG.xdmf").__str__()

    wss = XDMFFile(MPI.comm_world, wss_path)
    twssg = XDMFFile(MPI.comm_world, twssg_path)

    for f in [wss, twssg]:
        f.parameters["flush_output"] = True
        f.parameters["functions_share_mesh"] = True
        f.parameters["rewrite_function_mesh"] = False

    wss.write_checkpoint(WSS_abs, "WSS_abs", 0)
    twssg.write_checkpoint(TWSSG, "TWWSSG", 0)


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
    assert save_deg == 1, "This script only works for save_deg = 2"

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

    compute_hemodyanamics(visualization_separate_domain_folder, args.mesh_path, nu)

