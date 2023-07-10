import numpy as np
import h5py
import os

from dolfin import *
from postprocessing_common import read_command_line_stress


def compute_volume(case_path, mesh_name, dt, stride, save_deg):
    # Path to the 
    visualization_separate_domain_path = os.path.join(case_path, "Visualization_separate_domain")
    # Path to displacement file
    file_path_d = os.path.join(visualization_separate_domain_path, "d.h5")
    
    # get solid-only version of the mesh
    mesh_name = mesh_name + ".h5"
    mesh_name = mesh_name.replace(".h5","_solid_only.h5")
    mesh_path = os.path.join(case_path, "mesh", mesh_name)

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        if MPI.rank(MPI.comm_world) == 0:
            print("Warning, stress results are compromised by using save_deg = 1, especially using a coarse mesh. Recommend using save_deg = 2 instead for computing stress")
        mesh_path_viz = mesh_path
    else:
        mesh_path_viz = mesh_path.replace("_solid_only.h5","_refined_solid_only.h5")
    
    mesh_path = Path(mesh_path)
    mesh_path_viz = Path(mesh_path_viz)
        
    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    # Read refined mesh saved as HDF5 format
    mesh_viz = Mesh()
    with HDF5File(MPI.comm_world, mesh_path_viz.__str__(), "r") as mesh_file:
        mesh_file.read(mesh_viz, "mesh", False)

    if MPI.rank(MPI.comm_world) == 0:
        print("Define function spaces and functions")

    # Create higher-order function space for d, v and p
    dve = VectorElement('CG', mesh.ufl_cell(), save_deg)
    FSdv = FunctionSpace(mesh, dve)   # Higher degree FunctionSpace for d and v

    # Create visualization function space for d, v and p
    dve_viz = VectorElement('CG', mesh_viz.ufl_cell(), 1)
    FSdv_viz = FunctionSpace(mesh_viz, dve_viz)   # Visualisation FunctionSpace for d and v

    # Create higher-order function on unrefined mesh for post-processing calculations
    d = Function(FSdv)

    # Create lower-order function for visualization on refined mesh
    d_viz = Function(FSdv_viz)
    
    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    dv_trans = PETScDMCollection.create_transfer_matrix(FSdv_viz,FSdv)

    dx = Measure("dx")
    
    
    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    file_counter = 0 # Index of first time step
    file_1 = 1 # Index of second time step

    f = HDF5File(MPI.comm_world, file_path_d.__str__(), "r")
    vec_name = "/displacement/vector_%d" % file_counter
    t_0 = f.attributes(vec_name)["timestamp"]
    vec_name = "/displacement/vector_%d" % file_1
    t_1 = f.attributes(vec_name)["timestamp"]  
    time_between_files = t_1 - t_0
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation

    while True:
        try:
            f = HDF5File(MPI.comm_world, file_path_d.__str__(), "r")
            vec_name = "/displacement/vector_%d" % file_counter
            t = f.attributes(vec_name)["timestamp"]
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Timestep: {}".format(t), "=" * 10)
            f.read(d_viz, vec_name)
        except:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break        

        # Calculate d in P2 based on visualization refined P1
        d.vector()[:] = dv_trans*d_viz.vector()


    
        # Name function
        ep.rename("InfinitesimalStrain", "ep")
        sig.rename("TrueStress", "sig")
        ep_P.rename("MaximumPrincipalStrain", "ep_P")
        sig_P.rename("MaximumPrincipalStress", "sig_P")

        if MPI.rank(MPI.comm_world) == 0:
            print("Writing Additional Viz Files for Stresses and Strains!")

        # Write results
        ep_file.write(ep, t)
        sig_file.write(sig, t)
        ep_P_file.write(ep_P, t)
        sig_P_file.write(sig_P, t)

        # Update file_counter
        file_counter += stride


if __name__ == '__main__':
    folder, mesh, E_s, nu_s, dt, stride, save_deg = read_command_line_stress()
    compute_stress(folder,mesh, E_s, nu_s, dt, stride, save_deg)
