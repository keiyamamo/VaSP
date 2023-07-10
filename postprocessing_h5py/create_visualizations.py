import os
from glob import glob
import numpy as np
import postprocessing_common_h5py
#import spectrograms as spec
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 

"""
This script takes the visualization files from turtleFSI and outputs: 
(1) Domain-specific visualizations (wall displacement for the wall only, and velocity and pressure for the fluid only).
    This way the simualtion can be postprocessed as a CFD simulation and a solid Mechanics simulation separately. 
(2) Reduced save-deg visualization if save_deg = 2 (Creates a lightweight file for faster postprocessing)
(3) Band-Pass filtered visualizations for d, v, p and strain

A "Transformed Matrix" is created as well, which stores the output data in a format that can be opened quickly when we want to create spectrograms.

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    dt (float): Actual time step of simulation
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 
    dvp: postprocess d, v, or p (displacement velocity or pressure)
    bands: list of bands for band-pass filtering displacement. 

Example: --dt 0.000679285714286 --mesh file_case16_el06 --end_t 0.951

"""

def create_visualizations(case_path, mesh_name, save_deg, stride, ts, start_t, end_t, dvp, bands, perform_hi_pass):
    print(f"perform_hi_pass = {perform_hi_pass}")
    print("")

    # Get Mesh
    if save_deg == 1:
        mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
    else: 
        mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 
    mesh_path_sd1 = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
    mesh_path_fluid_sd1 = mesh_path_sd1.replace(".h5","_fluid_only.h5") # needed for mps
    mesh_path_solid_sd1 = mesh_path_sd1.replace(".h5","_solid_only.h5") # needed for mps

    # create path for output folders
    case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
    visualization_path = os.path.join(case_path, "Visualization")
    formatted_data_folder_name = "res_"+case_name+'_stride_'+str(stride)+"t"+str(start_t)+"_to_"+str(end_t)+"save_deg_"+str(save_deg)
    formatted_data_folder = os.path.join(case_path,formatted_data_folder_name)
    visualization_separate_domain_folder = os.path.join(case_path, "Visualization_separate_domain")
    visualization_hi_pass_folder = os.path.join(case_path, "Visualization_hi_pass")
    # NOTE: comment out for now as we do not need reduce_save_deg_viz 18/05/2023 Kei 
    # visualization_sd1_folder = os.path.join(case_path, "Visualization_sd1")
    
    # Create output folder and filenames for compressed data
    output_file_name = case_name+"_"+ dvp+"_mag.npz" 
    formatted_data_path = os.path.join(formatted_data_folder, output_file_name)
    
    # If the output file exists, don't re-make it
    if os.path.exists(formatted_data_path):
       print(f'Path: {formatted_data_path} found! Will not re-make npz files')
       time_between_input_files = postprocessing_common_h5py.get_time_between_files(visualization_path, dvp)

    elif dvp == "wss":
        print("Making compressed data:  Start time: {}, end time: {}".format(start_t,end_t))
        time_between_input_files = postprocessing_common_h5py.create_transformed_matrix(visualization_separate_domain_folder, formatted_data_folder, mesh_path_fluid_sd1, case_name, start_t,end_t, dvp, stride)

    elif dvp == "mps" or dvp == "strain":
        print("Making compressed data:  Start time: {}, end time: {}".format(start_t,end_t))
        time_between_input_files = postprocessing_common_h5py.create_transformed_matrix(visualization_separate_domain_folder, formatted_data_folder, mesh_path_solid_sd1, case_name, start_t,end_t, dvp, stride)

    else: 
        # Make the output h5 files with dvp magnitudes
        print("Making compressed data:  Start time: {}, end time: {}".format(start_t,end_t))
        time_between_input_files = postprocessing_common_h5py.create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path, case_name, start_t, end_t, dvp, stride)
    
    # Get the desired time between output files (reduce output frequency by "stride")
    time_between_output_files = time_between_input_files*stride 

    # create bands for hi-pass filter
    bands_list = bands.split(",")
    num_bands = int(len(bands_list)/2)
    lower_freq = np.zeros(num_bands)
    higher_freq = np.zeros(num_bands)
    for i in range(num_bands):
        lower_freq[i] = float(bands_list[2*i])
        higher_freq[i] = float(bands_list[2*i+1])

    if dvp != "wss" and dvp != "mps" and dvp != "strain":
        # Create the visualization files
        print("===Creating domain specific visualization files===")
        print("")
        postprocessing_common_h5py.create_domain_specific_viz(formatted_data_folder, visualization_separate_domain_folder, mesh_path, save_deg, time_between_output_files,start_t,dvp)

        # TODO: here mesh_path should be the mesh_path_sd1 ?
        # NOTE: For now, reduce_save_deg_viz is not necessary and skip it 18/05/2023 Kei 
        # postprocessing_common_h5py.reduce_save_deg_viz(formatted_data_folder, visualization_sd1_folder, mesh_path,1, time_between_output_files,start_t,dvp)

        #NOTE: I do not know why save_deg needs to be 1 here. It could be 2 as well? Also, do we need high pass viz for velocity and pressure?
        if dvp == "d" and perform_hi_pass:
            print("===creating hi pass viz===")
            print("")
            for i in range(len(lower_freq)):
                postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,start_t,dvp,lower_freq[i],higher_freq[i])
                postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,start_t,dvp,lower_freq[i],higher_freq[i],amplitude=True)
    
    elif dvp == "strain":
        for i in range(len(lower_freq)):
            postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path_solid_sd1,time_between_output_files,start_t,dvp,lower_freq[i],higher_freq[i])
            postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path_solid_sd1,time_between_output_files,start_t,dvp,lower_freq[i],higher_freq[i],amplitude=True)
    

if __name__ == '__main__':
    # Get Command Line Arguments and Input File Paths
    case_path, mesh_name, save_deg, stride, ts, start_t, end_t, dvp, bands, perform_hi_pass = postprocessing_common_h5py.read_command_line()
    create_visualizations(case_path, mesh_name, save_deg, stride, ts, start_t, end_t, dvp, bands, perform_hi_pass)
