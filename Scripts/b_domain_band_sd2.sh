#!/bin/bash
# Job name:
#SBATCH --job-name=b_domain_band_sd2
#
# Max running time (DD-HH:MM:SS)
#SBATCH --time=0-4:00:00 
#
# Project:
#SBATCH --account=nn9249k

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=1  # make sure that it is the same as "OMP_NUM_THREADS"
#SBATCH --output=/cluster/home/keiya/log/%x-%j.txt

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
config_file=$1
source $config_file
echo "Sourcing config file: $config_file"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, end time of simulation: $end_t"

source /cluster/shared/fenics/conf/fenics-2019.2.0.dev0.saga.foss-2022a-py3.10.conf

# Run postprocessing scripts
str="Running h5py postprocessing scripts!"
echo $str
# python -u $workflow_location/postprocessing_h5py/create_visualizations.py --case=$case_path --mesh=$mesh_path --dt=$dt --start_t=$start_t_sd2 --end_t=$end_t --dvp=v --save_deg=2 --stride=$stride_sd2 --bands=$bands
python -u $workflow_location/postprocessing_h5py/create_visualizations.py --case=$case_path --mesh=$mesh_path --dt=$dt --start_t=$start_t_sd2 --end_t=$end_t --dvp=d --save_deg=2 --stride=$stride_sd2 --bands=$bands --$hi_pass
# python -u $workflow_location/postprocessing_h5py/create_visualizations.py --case=$case_path --mesh=$mesh_path --dt=$dt --start_t=$start_t_sd2 --end_t=$end_t --dvp=p --save_deg=2 --stride=$stride_sd2 --bands=$bands