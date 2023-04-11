#!/bin/bash

source /cluster/home/keiya/Aneurysm_Workflow_FSI/script_params.config

if [ $operating_sys != "local" ] ;
then
cd $SLURM_SUBMIT_DIR
source /cluster/shared/fenics/conf/fenics-2019.1.0.saga.intel.conf
echo "Running scripts on Saga"
else
echo "Running scripts on local os"
fi

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt,  end time of simulation: $end_t"

# Run postprocessing scripts
str="Creating readable h5 files (must be run in serial)"
echo $str
python $workflow_location/postprocessing_fenics/compute_norms.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=1 --stride=2