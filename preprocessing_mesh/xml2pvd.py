from argparse import ArgumentParser
from dolfin import Mesh, MeshFunction, File
from pathlib import Path

def xml2pvd(mesh_file: str) -> None:
    """Take xml mesh file and convert to pvd file.

    Args: mesh_file (str): h5 mesh file
    Retuns: None
    """
    mesh = Mesh(mesh_file)
        
    # Rescale the mesh coordinated from [mm] to [m]
    x = mesh.coordinates()
    scaling_factor = 0.001  # from mm to m
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array()+1)  # FIX ME, THIS IS NOT NORMAL!


    mesh_file_path = Path(mesh_file)

    boudaries_file = mesh_file_path.parent / (mesh_file_path.stem + "_boundaries.pvd")
    ff = File(str(boudaries_file))
    ff << boundaries

    domains = MeshFunction("size_t", mesh, 3, mesh.domains())
    domains.set_values(domains.array()+1)  # in order to have fluid==1 and solid==2

    domains_file = mesh_file_path.parent / (mesh_file_path.stem + "_domains.pvd")
    ff = File(str(domains_file))
    ff << domains  
    print("Converted to pvd files")
    return None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--case', type=str, help="path to the file", metavar="PATH")
    args = parser.parse_args()
    xml2pvd(args.case)