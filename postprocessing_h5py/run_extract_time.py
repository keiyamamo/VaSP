import postprocessing_common_h5py as pc
import os

def main():
    case_path, mesh_name, save_deg, stride, dt, start_t, end_t, dvp, _, _ = pc.read_command_line()

    for folder in os.listdir(case_path):
    # check if the folder name starts with "res_"
        if folder.startswith("res_"):
            formatted_data_folder_name = os.path.join(case_path, folder)
            break

    formatted_data_folder = os.path.join(case_path,formatted_data_folder_name)
    output_folder = os.path.join(case_path, "time_extracted")

    if save_deg == 1:
        mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
    else: 
        mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 

    pc.extract_time_steps(formatted_data_folder, output_folder, start_t, end_t, stride, dvp, save_deg, mesh_path, dt)

    return None

if __name__ == '__main__':
    main()

