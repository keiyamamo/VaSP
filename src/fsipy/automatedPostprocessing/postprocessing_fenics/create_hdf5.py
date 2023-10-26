import numpy as np
import h5py
import re
from pathlib import Path
import json
import logging

from postprocessing_common import read_command_line
from dolfin import Mesh, HDF5File, VectorFunctionSpace, FunctionSpace, Function, MPI, parameters

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["reorder_dofs_serial"] = False


def create_hdf5(visualization_path, mesh_path, save_time_step, stride, start_t, end_t, extract_solid_only):

    """
    Loads displacement and velocity data directly from turtleFSI output (Visualization/displacement.h5, Visualization/velocity.h5, ) 
    and reformats the data so that it can be read easily in fenics (Visualization_Separate_Domain/d.h5 and Visualization_Separate_Domain/v.h5). 
    This script works with restarts as well, by using the .xdmf file to point to the correct h5 file at the correct time. 
    This script must be run in serial while the more compuattionally intensive operations (wss and stress calculations) can then be run in parallel.

    Args:
        visualization_path (Path): Path to the folder containing the visualization files (displacement.h5, velocity.h5, etc.)
        mesh_path (Path): Path to the mesh file (mesh.h5 or mesh_refined.h5 depending on save_deg)
        dt (float): Time step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (Separate Domain Visualization in this script)
        start_t (float): desired start time for the output file
        end_t (float): desired end time for the output file
        extracct_solid_only (bool): If True, only the solid domain is extracted for displacement. If False, both the fluid and solid domains are extracted.
    """

    # Define mesh path related variables
    fluid_domain_path = mesh_path.with_name(mesh_path.stem + "_fluid.h5")
    solid_domain_path = mesh_path.with_name(mesh_path.stem + "_solid.h5")

    # Check if the input mesh exists
    if not fluid_domain_path.exists() or not solid_domain_path.exists():
        raise ValueError("Mesh file not found.")

    # Read fluid and solid mesh
    logging.info("--- Reading fluid and solid mesh files \n")
    mesh_fluid = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_domain_path), "r") as mesh_file:
        mesh_file.read(mesh_fluid, "mesh", False)
    
      # Read refined solid mesh saved as HDF5 format
    messh_solid = Mesh()
    with HDF5File(MPI.comm_world, str(solid_domain_path), "r") as mesh_file:
        mesh_file.read(messh_solid, "mesh", False)
    
    # Define function spaces and functions
    logging.info("--- Defining function spaces and functions")
    Vf = VectorFunctionSpace(mesh_fluid, "CG", 1) # Velocity function space
    Vs = VectorFunctionSpace(messh_solid, "CG", 1) # Displacement function space
    v = Function(Vf) # Velocity function
    d = Function(Vs) # Displacement function

    # Define paths for velocity and displacement files
    xdmf_file_v = visualization_path / "velocity.xdmf" # Use velocity xdmf to determine which h5 file contains each timestep
    xdmf_file_d = visualization_path / "displacement.xdmf" # Use displacement xdmf to determine which h5 file contains each timestep

    # Get information about h5 files associated with xdmf files and also information about the timesteps
    logging.info("--- Getting information about h5 files")
    h5file_name_list, timevalue_list, index_list = output_file_lists(xdmf_file_v)
    h5file_name_list_d, _, index_list_d = output_file_lists(xdmf_file_d)

    fluidIDs, solidIDs, allIDs = get_domain_ids(mesh_path) # Get list of all nodes in fluid, solid domains

    # Remove this if statement since it can be done when we are using d_ids
    if extract_solid_only:
        logging.info("--- Extracting solid domain only")
        d_ids = solidIDs # Solid domain only
    else:
        d_ids = allIDs # Fluid and solid domain

    # Open up the first velocity.h5 file to get the number of timesteps and nodes for the output data
    # NOTE: not sure if we need this
    file = visualization_path / h5file_name_list[0]
    vectorData = h5py.File(str(file))
    vectorArray = vectorData['VisualisationVector/0'][fluidIDs,:] 
    # Open up the first displacement.h5 file to get the number of timesteps and nodes for the output data
    file_d = visualization_path / h5file_name_list_d[0]
    vectorData_d = h5py.File(str(file_d)) 
    vectorArray_d = vectorData['VisualisationVector/0'][d_ids,:] 

    # Deinfe path to the output files
    u_output_path = visualization_path / "u.h5"
    d_output_path = visualization_path / "d.h5"

    # Start file counter
    file_counter = 0 
    # Initialize variables
    h5_file_prev = ""
    h5_file_prev_d = ""

    # NOTE: while true is not a good idea, need to implement a better way to do this
    while True:

        try:

            time_file = timevalue_list[file_counter] # Current time
            print("=" * 10, "Timestep: {}".format(time_file), "=" * 10)

            # NOTE: this only happens if turtleFSI was used with save_step > 1. Not sure if we need to implement this
            if file_counter>0:
                if np.abs(time_file-timevalue_list[file_counter-1] - save_time_step) > 1e-8: # if the spacing between files is not equal to the intended timestep
                    print('Warning: Uenven temporal spacing detected!!')
            
            if time_file>=start_t and time_file <= end_t: # For desired time range:

                # Open input velocity h5 file
                h5_file = visualization_path / h5file_name_list[file_counter]
                if h5_file != h5_file_prev: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
                    vectorData.close()
                    vectorData = h5py.File(str(h5_file))
                h5_file_prev = h5_file # Record h5 file name for this step
    
                # Open input displacement h5 file
                h5_file_d = visualization_path / h5file_name_list_d[file_counter]
                if h5_file_d != h5_file_prev_d: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
                    vectorData_d.close()
                    vectorData_d = h5py.File(str(h5_file_d))
                h5_file_prev_d = h5_file_d # Record h5 file name for this step
    
                # Open up Vector Arrays from h5 file
                ArrayName = 'VisualisationVector/' + str((index_list[file_counter]))    
                vectorArrayFull = vectorData[ArrayName][:,:] # Important not to take slices of this array, slows code considerably... 
                ArrayName_d = 'VisualisationVector/' + str((index_list_d[file_counter]))    
                vectorArrayFull_d = vectorData_d[ArrayName_d][:,:] # Important not to take slices of this array, slows code considerably... 
                # instead make a copy (VectorArrayFull) and slice that.
                
                vectorArray = vectorArrayFull[fluidIDs, :]    
                vectorArray_d = vectorArrayFull_d[d_ids, :]    
                
                # Velocity
                vector_np_flat = vectorArray.flatten('F')
                v.vector().set_local(vector_np_flat)  # Set u vector
                print("Saved data in v.h5")
    
                # Displacement
                vector_np_flat_d = vectorArray_d.flatten('F')
                d.vector().set_local(vector_np_flat_d)  # Set d vector
                print("Saved data in d.h5")
    
                file_mode = "a" if file_counter > 0 else "w"
        
                # Save velocity
                viz_v_file = HDF5File(MPI.comm_world, str(u_output_path), file_mode=file_mode)
                viz_v_file.write(v, "/velocity", time_file)
                viz_v_file.close()
        
                # Save displacment
                viz_d_file = HDF5File(MPI.comm_world, str(d_output_path), file_mode=file_mode)
                viz_d_file.write(d, "/displacement", time_file)
                viz_d_file.close()


        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero
            print("=" * 10, "Finished reading solutions", "=" * 10)
            break

        # Update file_counter
        file_counter += stride
        #t += time_between_files*stride

# Helper Functions

# This function can be removed
def get_domain_topology(meshFile):
    # This function obtains the topology for the fluid, solid, and all elements of the input mesh
    # Importantly, it is ASSUMED that the fluid domain is labeled 1 and the solid domain is labeled 2 (and above, if multiple solid regions) 
    vectorData = h5py.File(meshFile,"r")
    domainsLoc = 'domains/values'
    domains = vectorData[domainsLoc][:] # Open domain array
    id_wall = (domains>1).nonzero() # domain = 2 and above is the solid
    id_fluid = (domains==1).nonzero() # domain = 1 is the fluid

    topologyLoc = 'domains/topology'
    allTopology = vectorData[topologyLoc][:,:] 
    wallTopology=allTopology[id_wall,:] 
    fluidTopology=allTopology[id_fluid,:]

    return fluidTopology, wallTopology, allTopology

# NOTE: this function could be removed
def get_domain_ids(meshFile):
    # This function obtains a list of the node IDs for the fluid, solid, and all elements of the input mesh

    # Get topology of fluid, solid and whole mesh
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    solidIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    return fluidIDs, solidIDs, allIDs

# NOTE: this function is necessary
def output_file_lists(xdmf_file):
    """
    If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    This loop determines the file names from the xdmf output file

    Args:
        xdmf_file (Path): Path to xdmf file

    Returns:
        h5file_name_list (list): List of names of h5 files associated with each timestep
        timevalue_list (list): List of time values in xdmf file
        index_list (list): List of indices of each timestp in the corresponding h5 file
    """

    file1 = open(xdmf_file, 'r') 
    Lines = file1.readlines() 
    h5file_name_list = []
    timevalue_list = []
    index_list = []
    
    # This loop goes through the xdmf output file and gets the time value (timevalue_list), associated 
    # .h5 file (h5file_name_list) and index of each timestep in the corresponding h5 file (index_list)
    for line in Lines: 
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            timevalue_list.append(time)

        elif 'VisualisationVector' in line:
            h5_pattern = '"HDF">(.+?):/'
            h5_str = re.findall(h5_pattern, line)
            h5file_name_list.append(h5_str[0])

            index_pattern = "VisualisationVector/(.+?)</DataItem>"
            index_str = re.findall(index_pattern, line)
            index = int(index_str[0])
            index_list.append(index)

    return h5file_name_list, timevalue_list, index_list


def main() -> None:

    assert MPI.size(MPI.comm_world) == 1, "This script only runs in serial."

    args = read_command_line()

    logging.basicConfig(level=20, format="%(message)s")

    # Define paths for visulization and mesh files
    folder_path = Path(args.folder)
    visualization_path = folder_path / "Visualization"

    # Read parameters from default_variables.json
    parameter_path = folder_path / "Checkpoint" / "default_variables.json"
    with open(parameter_path, "r") as f:
        parameters = json.load(f)
        save_deg = parameters["save_deg"]
        dt = parameters["dt"]
        save_step = parameters["save_step"]
        save_time_step = dt * save_step
        print("save_time_step: ", save_time_step)
    
    if save_deg == 2:
        mesh_path = folder_path / "Mesh" / "mesh_refined.h5"
        logging.info("--- Using refined mesh")
        assert mesh_path.exists(), "Mesh file not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        logging.info("--- Using non-refined mesh")
        assert mesh_path.exists(), "Mesh file not found."

    create_hdf5(visualization_path, mesh_path, save_time_step, args.stride, args.start_t, args.end_t, args.extract_solid_only)

if __name__ == '__main__':
    main()
