from pathlib import Path

from dolfin import *
from turtleFSI.modules import common
import os

from postprocessing_common import read_command_line

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["reorder_dofs_serial"] = False


def compute_stress(visualization_separate_domain_folder, mesh_path, stride):

    """
    Loads displacement fields from completed FSI simulation,
    and computes and saves the following solid mechanical quantities:
    (1) True Stress
    (2) Infinitesimal Strain
    (3) Maximum Principal Stress (True)
    (4) Maximum Principal Strain (Infinitesimal)
    edit June 19th, 2023:  we now read material properties from a "logfile" in the simulation directory or from "material_properties.txt"
    This script can now compute stress for subdomains with different material properties and different material models (Mooney-Rivlin, for example)

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        dt (float): Actual ime step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)

    """
    try:
        file_path_d = visualization_separate_domain_folder / "d_solid.h5"
        assert file_path_d.exists()
        solid_only = True
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using d_solid.h5 file \n")
    except AssertionError:
        file_path_d = visualization_separate_domain_folder / "d.h5"
        assert file_path_d.exists()
        solid_only = False
        if MPI.rank(MPI.comm_world) == 0:
            print("--- displacement is for the entire domain \n")

    file_d = HDF5File(MPI.comm_world, str(file_path_d), "r")

    with HDF5File(MPI.comm_world, str(file_path_d), "r") as f:
        dataset = get_dataset_names(f, step=stride, vector_filename="/displacement/vector_%d")

    # Read the original mesh and also the refined mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Read the original mesh and also the refined mesh \n")

    solid_mesh_path = mesh_path / "mesh_solid.h5" if solid_only else mesh_path / "mesh.h5"
    mesh = Mesh()
    with HDF5File(MPI.comm_world, str(solid_mesh_path), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    refined_mesh_path = mesh_path / "mesh_refined_solid.h5" if solid_only else mesh_path / "mesh_refined.h5"
    refined_mesh = Mesh()
    with HDF5File(MPI.comm_world, str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(refined_mesh, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")


    # _, domains, _ = get_mesh_domain_and_boundaries(mesh_path)

    print_MPI("Define function spaces and functions")

    # Create function space for the displacement on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    # Create function space for the displacement on the refined mesh with P2 elements
    Vv_non_refined = VectorFunctionSpace(mesh, "CG", 2)

    # Create visualization function space for d, v and p
    dve_viz = VectorElement('CG', mesh_viz.ufl_cell(), 1)
    FSdv_viz = FunctionSpace(mesh_viz, dve_viz)   # Visualisation FunctionSpace for d and v

    # Create higher-order function on unrefined mesh for post-processing calculations
    d = Function(FSdv)

    # Create lower-order function for visualization on refined mesh
    d_viz = Function(FSdv_viz)
    
    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    dv_trans = PETScDMCollection.create_transfer_matrix(FSdv_viz,FSdv)

    # Set up dx (dx_s for solid, dx_f for fluid) for each domain
    dx = Measure("dx", subdomain_data=domains)
    dx_s = {}
    dx_s_id_list = []
    for idx, solid_region in enumerate(solid_properties):
        dx_s_id = solid_region["dx_s_id"]
        dx_s[idx] = dx(dx_s_id, subdomain_data=domains) # Create dx_s for each solid region
        dx_s_id_list.append(dx_s_id)
        print_MPI(solid_region)

    dx_f = {}
    dx_f_id_list = []
    for idx, fluid_region in enumerate(fluid_properties):
        dx_f_id = fluid_region["dx_f_id"]
        dx_f[idx] = dx(dx_f_id, subdomain_data=domains) # Create dx_s for each solid region
        dx_f_id_list.append(dx_f_id)
        print_MPI(fluid_region)


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

    # NOTE: I guess sig is sigma and ep is epsilon, sig_P is sigma principal and ep_P is epsilon principal
    sig_path = (visualization_separate_domain_path / "TrueStress.xdmf").__str__()
    ep_path = (visualization_separate_domain_path / "InfinitesimalStrain.xdmf").__str__()
    sig_P_path = (visualization_separate_domain_path / "MaxPrincipalStress.xdmf").__str__()
    ep_P_path = (visualization_separate_domain_path / "MaxPrincipalStrain.xdmf").__str__()
    
    sig_file = XDMFFile(MPI.comm_world, sig_path)
    ep_file = XDMFFile(MPI.comm_world, ep_path)
    sig_P_file = XDMFFile(MPI.comm_world, sig_P_path)
    ep_P_file = XDMFFile(MPI.comm_world, ep_P_path)

    sig_file.parameters["rewrite_function_mesh"] = False
    ep_file.parameters["rewrite_function_mesh"] = False
    sig_P_file.parameters["rewrite_function_mesh"] = False
    ep_P_file.parameters["rewrite_function_mesh"] = False

    print_MPI("========== Start post processing ==========")


    # NOTE: Instead of using while True, we can use vampy get_datasets function like I did with create_seprate_domain_visualization.py
    while True:
        try:
            f = HDF5File(MPI.comm_world, file_path_d.__str__(), "r")
            vec_name = "/displacement/vector_%d" % file_counter
            t = f.attributes(vec_name)["timestamp"]
            print_MPI("========== Timestep: {} ==========".format(t))
            f.read(d_viz, vec_name)
        except Exception as error:
            print_MPI(error) # An exception occurred

            print_MPI("========== Finished reading solutions ==========")
            break        
        
        d_viz.rename("Displacement_test", "d_viz")

        # Calculate d in P2 based on visualization refined P1
        d.vector()[:] = dv_trans*d_viz.vector()

        # Deformation Gradient and first Piola-Kirchoff stress (PK1)
        deformationF = common.F_(d) # calculate deformation gradient from displacement
        
        # Cauchy (True) Stress and Infinitesimal Strain (Only accurate for small strains, if other strain desired, check)
        epsilon = common.eps(d) # Form for Infinitesimal strain (need polar decomposition if we want to calculate logarithmic/Hencky strain)


        # These two loops project the material equations for the fluid and solid domains onto the proper subdomains.
        # This type of projection requires more lines of code but is faster than the built-in "project()" function. 
        # All quantities defined below are linear, bilinear forms and trial and test functions for the tensor and scalar quantities computed
        a = 0 
        a_scal = 0
        L_sig = 0
        L_sig_P = 0
        L_ep = 0
        L_ep_P = 0
        v = TestFunction(Tens) 
        u = TrialFunction(Tens)
        v_scal = TestFunction(Scal) 
        u_scal = TestFunction(Scal) 

        for solid_region in range(len(dx_s_id_list)):

            PiolaKirchoff2 = common.S(d, solid_properties[solid_region]) # Form for second PK stress (using specified material model)
            sigma = (1/common.J_(d))*deformationF*PiolaKirchoff2*deformationF.T  # Form for Cauchy (true) stress 
            a+=inner(u,v)*dx_s[solid_region] # bilinear form
            a_scal+=inner(u_scal,v_scal)*dx_s[solid_region] # bilinear form
            L_sig+=inner(sigma,v)*dx_s[solid_region] # linear form
            L_ep+=inner(epsilon,v)*dx_s[solid_region]  # linear form

        for fluid_region in range(len(dx_f_id_list)):

            nought_value = 1e-10 # Value for stress components in fluid regions 
            sigma_nought = as_tensor([[nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value]]) # Add placeholder value to fluid region
            epsilon_nought = as_tensor([[nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value]]) # Add placeholder value to fluid region            
            a+=inner(u,v)*dx_f[fluid_region] # bilinear form
            a_scal+=inner(u_scal,v_scal)*dx_f[fluid_region] # bilinear form
            L_sig+=inner(sigma_nought,v)*dx_f[fluid_region]  # linear form
            L_ep+=inner(epsilon_nought,v)*dx_f[fluid_region] # linear form

        sig = solve_stress_forms(a,L_sig,Tens) # Calculate stress tensor 
        ep = solve_stress_forms(a,L_ep,Tens) # Calculate stress tensor 

        eigStrain11,eigStrain22,eigStrain33 = common.get_eig(ep) # Calculate principal strain
        eigStress11,eigStress22,eigStress33  = common.get_eig(sig)  # Calculate principal stress
        ep_P=project_solid(eigStrain11,Scal,dx) # Project onto whole domain
        sig_P=project_solid(eigStress11,Scal,dx)  # Project onto whole domain

        # Name function
        ep.rename("InfinitesimalStrain", "ep")
        sig.rename("TrueStress", "sig")
        ep_P.rename("MaximumPrincipalStrain", "ep_P")
        sig_P.rename("MaximumPrincipalStress", "sig_P")

        print_MPI("Writing Additional Viz Files for Stresses and Strains!")

        # Write results
        ep_file.write(ep, t)
        sig_file.write(sig, t)
        ep_P_file.write(ep_P, t)
        sig_P_file.write(sig_P, t)

        # Update file_counter
        file_counter += stride


if __name__ == '__main__':
    folder, mesh, _, _, _, dt, stride, save_deg, _, _ = read_command_line()
    compute_stress(folder,mesh, dt, stride, save_deg)