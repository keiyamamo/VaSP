#!/bin/bash
# Job name:
#SBATCH --job-name=d_compute_stress_wss_local
#
# Max running time (DD-HH:MM:SS)
#SBATCH --time=0-01:00:00 
#
# Project:
#SBATCH --account=nn9249k

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1  # make sure that it is the same as "OMP_NUM_THREADS"
#SBATCH --output=/cluster/home/keiya/log/%x-%j.txt

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
source /cluster/shared/fenics/conf/fenics-2019.2.0.dev0.saga.intel-2020a-py3.8.conf

config_file=$1
source $config_file
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, end time of simulation: $end_t"

str="Running fenics postprocessing scripts!"
echo $str
python $workflow_location/postprocessing_fenics/compute_wss_fsi.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=1
python $workflow_location/postprocessing_fenics/compute_solid_stress.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=1
python $workflow_location/postprocessing_fenics/compute_flow_rate_fsi.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=1
