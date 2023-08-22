from fsipy.automatedPreprocessing.vmtkmeshgeneratorfsi import vmtkMeshGeneratorFsi
from vmtk import vmtkdistancetospheres
from morphman import vmtkscripts, write_polydata
from vtk import vtkPolyData
from dolfin import Mesh, MeshFunction, File, HDF5File
from pathlib import Path


# Global array names
distanceToSpheresArrayName = "DistanceToSpheres"
distanceToSpheresArrayNameSolid = "Thickness"


def distance_to_spheres_solid_thickness(surface: vtkPolyData, save_path: str,
                                        distance_offset: float = 0, distance_scale: float = 0.1,
                                        min_distance: float = 0.25, max_distance: float = 0.3) -> vtkPolyData:
    """
    Determines the solid thickness using vmtkdistancetospheres.

    Args:
        surface (vtkPolyData): Input surface model
        save_path (str): Location to store processed surface
        distance_offset (float): Offset added to the distances
        distance_scale (float): Scale applied to the distances
        min_distance (float): Minimum value for the distances
        max_distance (float): Maximum value for the distances

    Returns:
        surface (vtkPolyData): Processed surface model with info on solid thickness

    """
    distanceToSpheres = vmtkdistancetospheres.vmtkDistanceToSpheres()
    distanceToSpheres.Surface = surface
    distanceToSpheres.DistanceOffset = distance_offset
    distanceToSpheres.DistanceScale = distance_scale
    distanceToSpheres.MinDistance = min_distance
    distanceToSpheres.MaxDistance = max_distance
    distanceToSpheres.DistanceToSpheresArrayName = distanceToSpheresArrayNameSolid
    distanceToSpheres.Execute()
    distance_to_sphere = distanceToSpheres.Surface

    write_polydata(distance_to_sphere, save_path)

    return distance_to_sphere


def dist_sphere_spheres(surface: vtkPolyData, save_path: str,
                        distance_offset: float, distance_scale: float,
                        min_distance: float, max_distance: float) -> vtkPolyData:
    """
    Determines the target edge length for each cell on the surface, including
    potential refinement or coarsening of certain user specified areas.
    Level of refinement/coarseness is determined based on the distance to the spheres.

    Args:
        surface (vtkPolyData): Input surface model
        save_path (str): Location to store processed surface
        distance_offset (float): Offset added to the distances
        distance_scale (float): Scale applied to the distances
        min_distance (float): Minimum value for the distances
        max_distance (float): Maximum value for the distances

    Returns:
        surface (vtkPolyData): Processed surface model with info on cell specific target edge length
    """
    distanceToSpheres = vmtkdistancetospheres.vmtkDistanceToSpheres()
    distanceToSpheres.Surface = surface
    distanceToSpheres.DistanceOffset = distance_offset
    distanceToSpheres.DistanceScale = distance_scale
    distanceToSpheres.MinDistance = min_distance
    distanceToSpheres.MaxDistance = max_distance
    distanceToSpheres.DistanceToSpheresArrayName = distanceToSpheresArrayName
    distanceToSpheres.Execute()
    distance_to_sphere = distanceToSpheres.Surface

    surfaceCurvature = vmtkscripts.vmtkSurfaceCurvature()
    surfaceCurvature.AbsoluteCurvature = 1
    surfaceCurvature.MedianFiltering = 1
    surfaceCurvature.CurvatureType = "gaussian"
    surfaceCurvature.Offset = 0.15
    surfaceCurvature.BoundedReciprocal = 1
    surfaceCurvature.Surface = distance_to_sphere
    surfaceCurvature.Execute()
    distance_to_sphere = surfaceCurvature.Surface

    surfaceArrayOperation = vmtkscripts.vmtkSurfaceArrayOperation()
    surfaceArrayOperation.Surface = distance_to_sphere
    surfaceArrayOperation.InputArrayName = "Curvature"
    surfaceArrayOperation.Input2ArrayName = distanceToSpheresArrayName
    surfaceArrayOperation.ResultArrayName = "Size"
    surfaceArrayOperation.Operation = "multiply"
    surfaceArrayOperation.Execute()
    distance_to_sphere = surfaceArrayOperation.Surface

    write_polydata(distance_to_sphere, save_path)

    return distance_to_sphere


def generate_mesh(surface: vtkPolyData, number_of_sublayers_fluid: int, number_of_sublayers_solid: int,
                  solid_thickness: str, solid_thickness_parameters: list) -> tuple:
    """
    Generates a mesh suitable for FSI from an input surface model.

    Args:
        surface (vtkPolyData): Surface model to be meshed.
        number_of_sublayers_fluid (int): Number of sublayers for fluid.
        number_of_sublayers_solud (int): Number of sublayers for solid.
        solid_thickness (str): Type of solid thickness ('variable' or 'constant').
        solid_thickness_parameters (list): List of parameters for solid thickness.

    Returns:
        tuple: A tuple containing the generated mesh (vtkUnstructuredGrid) and the remeshed surface (vtkPolyData).
    """
    print("--- Creating FSI mesh")

    meshGenerator = vmtkMeshGeneratorFsi()
    meshGenerator.Surface = surface

    # Mesh Parameters
    meshGenerator.ElementSizeMode = "edgelengtharray"  # Variable size mesh
    meshGenerator.TargetEdgeLengthArrayName = "Size"  # Variable size mesh
    meshGenerator.LogOn = 1
    meshGenerator.BoundaryLayer = 1
    meshGenerator.NumberOfSubLayersSolid = number_of_sublayers_fluid
    meshGenerator.NumberOfSubLayersFluid = number_of_sublayers_solid
    meshGenerator.BoundaryLayerOnCaps = 0
    meshGenerator.SubLayerRatioFluid = 0.75
    meshGenerator.SubLayerRatioSolid = 0.75
    meshGenerator.BoundaryLayerThicknessFactor = 0.5
    meshGenerator.Tetrahedralize = 1
    meshGenerator.VolumeElementScaleFactor = 0.8
    meshGenerator.EndcapsEdgeLengthFactor = 1.0

    # Solid thickness handling
    if solid_thickness == 'variable':
        meshGenerator.ElementSizeModeSolid = "edgelengtharray"
        meshGenerator.TargetEdgeLengthArrayNameSolid = distanceToSpheresArrayNameSolid
    else:
        meshGenerator.ElementSizeModeSolid = "edgelength"
        meshGenerator.SolidThickness = solid_thickness_parameters[0]

    # IDs
    meshGenerator.SolidSideWallId = 11
    meshGenerator.InterfaceId_fsi = 22
    meshGenerator.InterfaceId_outer = 33
    meshGenerator.VolumeId_fluid = 0  # (keep to 0)
    meshGenerator.VolumeId_solid = 1

    # Generate mesh
    meshGenerator.Execute()
    remeshed_surface = meshGenerator.RemeshedSurface
    generated_mesh = meshGenerator.Mesh

    return generated_mesh, remeshed_surface


def convert_xml_mesh_to_hdf5(file_name_xml_mesh: str, scaling_factor: float = 0.001) -> None:
    """Converts an XML mesh to an HDF5 mesh.

    Args:
        file_name_xml_mesh (str): The name of the XML mesh file.
        scaling_factor (float, optional): A scaling factor to apply to the mesh coordinates.
                                          The default value is 0.001, which converts from millimeters to meters.

    Returns:
        None

    Raises:
        FileNotFoundError: If the XML mesh file does not exist.
    """

    # Check if the XML mesh file exists
    xml_mesh_path = Path(file_name_xml_mesh)
    if not xml_mesh_path.is_file():
        raise FileNotFoundError(f"The file '{xml_mesh_path}' does not exist.")

    mesh = Mesh(str(xml_mesh_path))

    # Rescale the mesh coordinates
    x = mesh.coordinates()
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array() + 1)  # FIXME: Explain why this is necessary

    base, first_dot, rest = xml_mesh_path.name.partition('.')
    file_name_boundaries = str(xml_mesh_path.with_name(base + "_boundaries.pvd"))
    boundary_file = File(file_name_boundaries)
    boundary_file << boundaries

    domains = MeshFunction("size_t", mesh, 3, mesh.domains())
    domains.set_values(domains.array() + 1)  # in order to have fluid==1 and solid==2

    file_name_domains = str(xml_mesh_path.with_name(base + "_domains.pvd"))
    domain_file = File(file_name_domains)
    domain_file << domains

    file_name_h5_mesh = str(xml_mesh_path.with_name(base + '.h5'))
    hdf = HDF5File(mesh.mpi_comm(), file_name_h5_mesh, "w")
    hdf.write(mesh, "/mesh")
    hdf.write(boundaries, "/boundaries")
    hdf.write(domains, "/domains")