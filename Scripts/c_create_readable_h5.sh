#!/bin/bash

source /cluster/home/keiya/Aneurysm_Workflow_FSI/script_params.config

source /cluster/shared/fenics/conf/fenics-2019.1.0.saga.intel.conf

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
source $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt,  end time of simulation: $end_t"

# Run postprocessing scripts
str="Creating readable h5 files (must be run in serial)"
echo $str
python $workflow_location/postprocessing_fenics/compute_readable_h5.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=1