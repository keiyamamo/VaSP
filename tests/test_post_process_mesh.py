import sys
from pathlib import Path
from io import StringIO

import pytest
import numpy as np
from dolfin import Mesh, cpp

from fsipy.automatedPreprocessing.automated_preprocessing import read_command_line, \
    run_pre_processing
from fsipy.automatedPostprocessing.postprocessing_mesh import create_refined_mesh

@pytest.fixture(scope="function")
def temporary_hdf5_file(tmpdir):
    """
    Fixture for generating a temporary HDF5 file path with a mesh for testing purposes.
    """
    # Define the path to the generated mesh
    original_model_path = Path("tests/test_data/artery/artery.stl")
    model_path = Path(tmpdir) / original_model_path.name
    mesh_path_hdf5 = model_path.with_suffix(".h5")

    # Make a copy of the original model
    model_path.write_text(original_model_path.read_text())

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            meshing_method="diameter",
            smoothing_method="taubin",
            refine_region=False,
            coarsening_factor=1.3,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=1,
            inlet_flow_extension_length=1,
        )
    )

    # Run pre processing to generate the mesh
    run_pre_processing(**common_input)

    yield mesh_path_hdf5  # Provide the temporary file path as a fixture

def test_refine_mesh(temporary_hdf5_file):
    """
    Test post processing mesh refinement.
    """
    # Get the mesh path from the temporary_hdf5_file fixture
    mesh_path = Path(temporary_hdf5_file)
    
