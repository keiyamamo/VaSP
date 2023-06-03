#!/bin/bash
# Job name:
#SBATCH --job-name=c_create_readable_h5
#
# Max running time (DD-HH:MM:SS)
#SBATCH --time=0-4:00:00 
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

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
source $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt,  end time of simulation: $end_t"

# Run postprocessing scripts
str="Creating readable h5 files (must be run in serial)"
echo $str
python $workflow_location/postprocessing_fenics/compute_readable_h5.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=1