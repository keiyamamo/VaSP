from pathlib import Path
import numpy as np
import h5py
from dolfin import *
import os
from postprocessing_common import read_command_line_stress
import stress_strain

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. Doesnt affect the speed
parameters["reorder_dofs_serial"] = False

def compute_strain(case_path, mesh_name, E_s, nu_s, dt, stride, save_deg):

    """
    Loads displacement fields from completed FSI simulation,
    and computes and saves the following solid mechanical quantities:
    
    (1) Green-Lagrangian Strain

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        E_s (float): Elastic Modulus
        nu_s (float): Poisson's Ratio
        dt (float): Actual ime step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)

    """
    visualization_separate_domain_path = os.path.join(case_path, "Visualization_separate_domain")
    file_path_d = os.path.join(case_path, "Visualization_separate_domain", "d.h5") # Displacement

    ep_path = os.path.join(visualization_separate_domain_path, "GreenLagrangianStrain.xdmf")

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
    
    # Create tensor function space for stress and strain (this is necessary to evaluate tensor valued functions)
    '''
    Strain/stress are in L2, therefore we use a discontinuous function space with a degree of 1 for P2P1 elements
    Could also use a degree = 0 to get a constant-stress representation in each element
    For more info see the Fenics Book (P62, or P514-515), or
    https://comet-fenics.readthedocs.io/en/latest/demo/viscoelasticity/linear_viscoelasticity.html?highlight=DG#A-mixed-approach
    https://fenicsproject.org/qa/10363/what-is-the-most-accurate-way-to-recover-the-stress-tensor/
    https://fenicsproject.discourse.group/t/why-use-dg-space-to-project-stress-strain/3768
    '''
    Te = TensorElement("DG", mesh.ufl_cell(), save_deg-1) 
    Tens = FunctionSpace(mesh, Te)
    Fe = FiniteElement("DG", mesh.ufl_cell(), save_deg-1) 
    Scal = FunctionSpace(mesh, Fe)

    
    ep_file = XDMFFile(MPI.comm_world, ep_path)    
    ep_file.parameters["rewrite_function_mesh"] = False

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

        # Deformation Gradient and first Piola-Kirchoff stress (PK1)        
        epsilon = stress_strain.E(d) # Form for Green-Lagrangian strain
        ep = stress_strain.project_solid(epsilon,Tens,dx) # Calculate stress tensor (this projection method is 6x faster than the built in version)
        # Name function
        ep.rename("GeenLagrangianStrain", "ep")
        
        if MPI.rank(MPI.comm_world) == 0 and file_counter % 100 == 0:
            print("Writing Additional Viz Files for Stresses and Strains! Time step: %d" % file_counter)

        # Write results
        ep_file.write(ep, t)

        # Update file_counter
        file_counter += stride


if __name__ == '__main__':
    folder, mesh, E_s, nu_s, dt, stride, save_deg = read_command_line_stress()
    compute_strain(folder,mesh, E_s, nu_s, dt, stride, save_deg)
