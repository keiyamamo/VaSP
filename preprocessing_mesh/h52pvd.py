from argparse import ArgumentParser
from dolfin import Mesh, HDF5File, MeshFunction, File
from pathlib import Path

def h52pvd(mesh_file: str) -> None:
    """Take h5 mesh file and convert to pvd file.

    Args: mesh_file (str): h5 mesh file
    Retuns: None
    """
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_file, "r")
    hdf.read(mesh, "/mesh", False)
    #NOTE: this is the test
    x = mesh.coordinates()
    scaling_factor = 1000  # from m to mm
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")
    # Convert to pvd
    output_filename = Path(mesh_file).with_suffix(".pvd")
    f = File(str(output_filename))
    f << boundaries
    f << domains
    
    return None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--case', type=str, help="path to the file", metavar="PATH")
    args = parser.parse_args()
    h52pvd(args.case)