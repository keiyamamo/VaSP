#!/bin/bash


module --quiet purge  # Reset the modules to the system default
source /cluster/shared/fenics/conf/fenics-2019.1.0.saga.intel.conf


# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt,  end time of region of interest: $end_t_ROI"

str="Running spectral/other postprocessing scripts!"
echo $str

python $workflow_location/postprocessing_h5py/create_spectrograms_chromagrams.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --start_t=$start_t_sd1 --save_deg=$save_deg --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere --stride=$stride_sd1 --dvp=d
python $workflow_location/postprocessing_h5py/create_spectrograms_chromagrams.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --start_t=$start_t_sd1 --save_deg=$save_deg --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere --stride=$stride_sd1 --dvp=v
python $workflow_location/postprocessing_h5py/create_spectrograms_chromagrams.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --start_t=$start_t_sd1 --save_deg=$save_deg --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere --stride=$stride_sd1 --dvp=p --p_spec_type="wall"
