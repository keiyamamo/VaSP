#!/bin/bash
# Job name:
#SBATCH --job-name=a_create_meshes
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
source /cluster/shared/fenics/conf/fenics-2019.1.0.saga.intel.conf
module list

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
config_file=$1
source $config_file
echo "config_file: $config_file"

# Run postprocessing scripts
str="Creating meshes for postprocessing!"
echo $str

# python $workflow_location/postprocessing_mesh/Create_Refined_Mesh.py --case=$case_path --mesh=$mesh_path
# python $workflow_location/postprocessing_mesh/Create_Solid_Only_Mesh.py --case=$case_path --mesh=$mesh_path
# python $workflow_location/postprocessing_mesh/Create_Fluid_Only_Mesh.py --case=$case_path --mesh=$mesh_path
# python $workflow_location/postprocessing_mesh/Create_Solid_Only_Mesh.py --case=$case_path --mesh=$refined_mesh_path
python -u $workflow_location/postprocessing_mesh/Create_Fluid_Only_Mesh.py --case=$case_path --mesh=$refined_mesh_path
