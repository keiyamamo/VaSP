from argparse import ArgumentParser
from dolfin import Mesh, HDF5File, MeshFunction, File
from os import path

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
    # Convert to pvd for domain and boundaries
    domain_file = path.splitext(mesh_file)[0] + "_domains.pvd"
    boundary_file = path.splitext(mesh_file)[0] + "_boundaries.pvd"
    f_domain = File(domain_file)
    f_boundary = File(boundary_file)
    f_boundary << boundaries
    f_domain << domains
    
    return None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--i', type=str, help="path to the file", metavar="PATH")
    args = parser.parse_args()
    h52pvd(args.i)