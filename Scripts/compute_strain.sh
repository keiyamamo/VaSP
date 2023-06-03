#!/bin/bash
# Job name:
#SBATCH --job-name=d_compute_stress_wss
#
# Max running time (DD-HH:MM:SS)
#SBATCH --time=0-02:00:00 
#
# Project:
#SBATCH --account=nn9249k

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1  # make sure that it is the same as "OMP_NUM_THREADS"
#SBATCH --output=/cluster/home/keiya/log/%x-%j.txt

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
source /cluster/shared/fenics/conf/fenics-2019.2.0.dev0.saga.foss-2022a-py3.10.conf
config_file=$1
source $config_file
echo "Sourcing config file: $config_file"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, end time of simulation: $end_t"

# srun python $workflow_location/postprocessing_fenics/compute_strain.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=$save_deg --stride=$stride_sd2
python $workflow_location/postprocessing_fenics/compute_strain.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=$save_deg --stride=$stride_sd2
