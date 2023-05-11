"""
When restarting a simulation, the visualization files are not always correct.
This scripts fixes the visualization files by checking mesh in h5 files
NOTE: This problem is not easy to fix with this script. This is just an attempt
"""
import common_meshing
import os
import h5py

def main():
    # Here folder is the "Visualization" folder
    folder, _ = common_meshing.read_command_line()

    # Here we find the path to the correct visualization file
    wrongNumberVizPath = os.path.join(folder, 'velocity_run_1.h5')
    correctNumberVizPath = os.path.join(folder, 'velocity.h5')
    # Open the files using h5py
    wrongNumberViz = h5py.File(wrongNumberVizPath)
    correctNumberViz = h5py.File(correctNumberVizPath)
    # Get the correct node numbering from the mesh
    correctNumberNodes = correctNumberViz['Mesh/0/mesh/geometry'][:]
    wrongNumberNodes = wrongNumberViz['Mesh/0/mesh/geometry'][:]

    correctTopology = correctNumberViz['Mesh/0/mesh/topology'][:]

    # Check if the node numbering is correct
    if (correctNumberNodes == wrongNumberNodes).all():
        print('Node numbering is correct')
    else:
        print('Node numbering is incorrect')
        # Copy the correct node numbering to the wrong visualization file
        wrongNumberViz['Mesh/0/mesh/geometry'][:] = correctNumberNodes
        wrongNumberViz['Mesh/0/mesh/topology'][:] = correctTopology
        print('Node numbering is now correct')
        # Save the file
        wrongNumberViz.close()

if __name__ == '__main__':
    main()