import numpy as np
import h5py
import re
from pathlib import Path

from postprocessing_common import read_command_line, get_time_between_files
from dolfin import Mesh, MeshFunction, HDF5File, VectorFunctionSpace, FunctionSpace, Function, MPI, File, parameters

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["reorder_dofs_serial"] = False


def create_hdf5(folder_path, mesh_path, dt, stride, save_deg, start_t, end_t):

    """
    Loads displacement and velocity data directly from turtleFSI output (Visualization/displacement.h5, Visualization/velocity.h5, ) 
    and reformats the data so that it can be read easily in fenics (Visualization_Separate_Domain/d.h5 and Visualization_Separate_Domain/v.h5). 
    This script works with restarts as well, by using the .xdmf file to point to the correct h5 file at the correct time. 
    This script must be run in serial while the more compuattionally intensive operations (wss and stress calculations) can then be run in parallel.

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        dt (float): Time step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (Separate Domain Visualization in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)
        start_t (float): desired start time for the output file
        end_t (float): desired end time for the output file

    """

    # Define mesh path related variables
    fluid_domain_path = mesh_path.with_name(mesh_path.stem + "_fluid.h5")
    solid_domain_path = mesh_path.with_name(mesh_path.stem + "_solid.h5")

    # Check if the input mesh exists
    if not fluid_domain_path.exists() or not solid_domain_path.exists():
        raise ValueError("Mesh file not found.")

    # Read fluid and solid mesh
    mesh_fluid = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_domain_path), "r") as mesh_file:
        mesh_file.read(mesh_fluid, "mesh", False)
    
      # Read refined solid mesh saved as HDF5 format
    messh_solid = Mesh()
    with HDF5File(MPI.comm_world, str(solid_domain_path), "r") as mesh_file:
        mesh_file.read(messh_solid, "mesh", False)
    
    # Define function spaces and functions
    Vf = VectorFunctionSpace(mesh_fluid, "CG", 1) # Velocity function space
    Vs = VectorFunctionSpace(messh_solid, "CG", 1) # Displacement function space
    v = Function(Vf) # Velocity function
    d = Function(Vs) # Displacement function

    # Get data from input mesh, .h5 and .xdmf files
    xdmf_file_v = folder_path / "Visualization" / "velocity.xdmf" # Use velocity xdmf to determine which h5 file contains each timestep
    xdmf_file_d = folder_path / "Visualization" / "displacement.xdmf" # Use displacement xdmf to determine which h5 file contains each timestep
    h5_ts, time_ts, index_ts = output_file_lists(xdmf_file_v) # Get list of h5 files containing each timestep, and corresponding indices for each timestep
    h5_ts_d, time_ts_d, index_ts_d = output_file_lists(xdmf_file_d)

    displacement_domains = "all"
    fluidIDs, solidIDs, allIDs = get_domain_ids(mesh_path) # Get list of all nodes in fluid, solid domains
    v_ids = fluidIDs # for v.h5, we only want the fluid ids, so we can use CFD postprocessing code
    if displacement_domains == "all": # for d.h5, we can choose between using the entire domain and just the solid domain
        d_ids = allIDs # Fluid and solid domain
    else:
        d_ids = solidIDs # Solid domain only

   
    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Initialize variables
    tol = 1e-8  # temporal spacing tolerance, if this tolerance is exceeded, a warning flag will indicate that the data has uneven temporal spacing
    h5_file_prev = ""
    h5_file_prev_d = ""

    # Start file counter
    file_counter = 0 
    t_0, time_between_files = get_time_between_files(xdmf_file_v)
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation

    # Open up the first velocity.h5 file to get the number of timesteps and nodes for the output data
    file = visualization_path + '/'+  h5_ts[0]
    vectorData = h5py.File(file) 
    vectorArray = vectorData['VisualisationVector/0'][v_ids,:] 
    # Open up the first displacement.h5 file to get the number of timesteps and nodes for the output data
    file_d = visualization_path + '/'+  h5_ts_d[0]
    vectorData_d = h5py.File(file_d) 
    vectorArray_d = vectorData['VisualisationVector/0'][d_ids,:] 

    while True:

        try:

            time_file = time_ts[file_counter] # Current time
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Timestep: {}".format(time_file), "=" * 10)
            
            if file_counter>0:
                if np.abs(time_file-time_ts[file_counter-1] - time_between_files) > tol: # if the spacing between files is not equal to the intended timestep
                    print('Warning: Uenven temporal spacing detected!!')
            
            if time_file>=start_t and time_file <= end_t: # For desired time range:

                # Open input velocity h5 file
                h5_file = visualization_path + '/'+h5_ts[file_counter]
                if h5_file != h5_file_prev: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
                    vectorData.close()
                    vectorData = h5py.File(h5_file) 
                h5_file_prev = h5_file # Record h5 file name for this step
    
                # Open input displacement h5 file
                h5_file_d = visualization_path + '/'+h5_ts_d[file_counter]
                if h5_file_d != h5_file_prev_d: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
                    vectorData_d.close()
                    vectorData_d = h5py.File(h5_file_d) 
                h5_file_prev_d = h5_file_d # Record h5 file name for this step
    
                # Open up Vector Arrays from h5 file
                ArrayName = 'VisualisationVector/' + str((index_ts[file_counter]))    
                vectorArrayFull = vectorData[ArrayName][:,:] # Important not to take slices of this array, slows code considerably... 
                ArrayName_d = 'VisualisationVector/' + str((index_ts_d[file_counter]))    
                vectorArrayFull_d = vectorData_d[ArrayName_d][:,:] # Important not to take slices of this array, slows code considerably... 
                # instead make a copy (VectorArrayFull) and slice that.
                
                vectorArray = vectorArrayFull[v_ids,:]    
                vectorArray_d = vectorArrayFull_d[d_ids,:]    
                
                # Velocity
                vector_np_flat = vectorArray.flatten('F')
                v_viz.vector().set_local(vector_np_flat)  # Set u vector
                if MPI.rank(MPI.comm_world) == 0:
                    print("Saved data in v.h5")
    
                # Displacement
                vector_np_flat_d = vectorArray_d.flatten('F')
                d_viz.vector().set_local(vector_np_flat_d)  # Set d vector
                if MPI.rank(MPI.comm_world) == 0:
                    print("Saved data in d.h5")
    
                file_mode = "w" if not os.path.exists(v_path_in) else "a"
        
                # Save velocity
                # NOTE: If we switch to using xdmf write_checkpoint, then we can get both
                # visualization and fenics readable data in one file. Depending on the
                # size of the data, this may be more efficient.
                viz_v_file = HDF5File(MPI.comm_world, v_path_in, file_mode=file_mode)
                viz_v_file.write(v_viz, "/velocity", time_file)
                viz_v_file.close()
        
                # Save displacment
                viz_d_file = HDF5File(MPI.comm_world, d_path_in, file_mode=file_mode)
                viz_d_file.write(d_viz, "/displacement", time_file)
                viz_d_file.close()


        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break

        # Update file_counter
        file_counter += stride
        #t += time_between_files*stride

# Helper Functions

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

def get_domain_ids(meshFile):
    # This function obtains a list of the node IDs for the fluid, solid, and all elements of the input mesh

    # Get topology of fluid, solid and whole mesh
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    solidIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    return fluidIDs, solidIDs, allIDs

def output_file_lists(xdmf_file):
    # If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    # This loop determines the file names from the xdmf output file
    file1 = open(xdmf_file, 'r') 
    Lines = file1.readlines() 
    h5_ts=[]
    time_ts=[]
    index_ts=[]
    
    # This loop goes through the xdmf output file and gets the time value (time_ts), associated 
    # .h5 file (h5_ts) and index of each timestep inthe corresponding h5 file (index_ts)
    for line in Lines: 
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            time_ts.append(time)

        elif 'VisualisationVector' in line:
            #print(line)
            h5_pattern = '"HDF">(.+?):/'
            h5_str = re.findall(h5_pattern, line)
            h5_ts.append(h5_str[0])

            index_pattern = "VisualisationVector/(.+?)</DataItem>"
            index_str = re.findall(index_pattern, line)
            index = int(index_str[0])
            index_ts.append(index)
    time_increment_between_files = time_ts[2] - time_ts[1] # Calculate the time between files from xdmf file

    return h5_ts, time_ts, index_ts


def main() -> None:
    assert MPI.size(MPI.comm_world) == 1, "This script only runs in serial."

    #TODO: read parameter file and check what save_deg was used.
    # If, save_deg = 2 was used, then use refined mesh as mesh path

    folder, mesh, _, _, _, dt, stride, save_deg, start_t, end_t = read_command_line()

    create_hdf5(folder, mesh, dt, stride, save_deg, start_t, end_t)

if __name__ == '__main__':
    main()
