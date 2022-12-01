from argparse import ArgumentParser
from dolfin import *

def xml2h5(mesh_name: str) -> None:
    """
    Take xml mesh file and convert to h5 file.

    Args: mesh_name (str): xml mesh file
    Retuns: None
    """

    mesh_file = Mesh("meshes/"+mesh_name+"_fsi.xml")

    # Rescale the mesh coordinated from [mm] to [m]
    # necessary due to vmtk 
    x = mesh_file.coordinates()
    scaling_factor = 0.001  # from mm to m
    x[:, :] *= scaling_factor
    mesh_file.bounding_box_tree().build(mesh_file)

    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh_file, 2, mesh_file.domains())

    boundaries.set_values(boundaries.array()+1)  # FIX ME, THIS IS NOT NORMAL!

    ff = File("meshes/"+mesh_name+"_boundaries.pvd")
    ff << boundaries

    domains = MeshFunction("size_t", mesh_file, 3, mesh_file.domains())
    domains.set_values(domains.array()+1)  # in order to have fluid==1 and solid==2

    ff = File("meshes/"+mesh_name+"_domains.pvd")
    ff << domains  

    hdf = HDF5File(mesh_file.mpi_comm(), "meshes/file_"+mesh_name+".h5", "w")
    hdf.write(mesh_file, "/mesh")
    hdf.write(boundaries, "/boundaries")
    hdf.write(domains, "/domains")

    print("Converted to h5 files")

    return None

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--mesh_name', type=str, help="input mesh name")
    args = parser.parse_args()
    xml2h5(args.mesh_name)