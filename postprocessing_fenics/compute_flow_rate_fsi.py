import numpy as np
import os
from pathlib import Path

from dolfin import HDF5File, Mesh, FunctionSpace, VectorElement, ds, assemble, inner, Constant, parameters, MPI, MeshFunction, Function, FacetNormal
from postprocessing_common import read_command_line, get_time_between_files

# set compiler arguments
parameters["reorder_dofs_serial"] = False

"""
Example usage:
python compute_flow_rate_fsi.py --case /cluster/work/users/keiya/case9_150k/ --mesh file_case9_el047 --save_deg 2
"""

def compute_flow_rate(case_path, mesh_name, stride, save_deg):

    # File paths
    visualization_path = os.path.join(case_path, "Visualization_separate_domain")
    file_path_flow_rate = os.path.join(visualization_path, "flow_rates.txt")
    file_path_u = os.path.join(visualization_path, "v.h5")

     # get fluid-only version of the mesh
    mesh_name = mesh_name + ".h5"
    mesh_path = os.path.join(case_path, "mesh", mesh_name)

    # Read mesh saved as HDF5 format
    # Here, we only want to extract the boundaries to create ds based on IDs
    whole_mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as whole_mesh_file:
        whole_mesh_file.read(whole_mesh, "mesh", False)
        boundaries = MeshFunction("size_t", whole_mesh, 2)
        whole_mesh_file.read(boundaries, "/boundaries")

    # Inlet/outlet differential
    dsi = ds(2, domain=whole_mesh, subdomain_data=boundaries)
    dso3 = ds(3, domain=whole_mesh, subdomain_data=boundaries)
    
    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg > 1:
        fluid_mesh_path = mesh_path.replace(".h5", "_refined_fluid_only.h5")
    
    # Read fluid mesh and define it as the fluid mesh
    fluid_mesh = Mesh()
    with HDF5File(MPI.comm_world, fluid_mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(fluid_mesh, "mesh", False)

    # Create function space defined on the fluid domain
    dve = VectorElement('CG', fluid_mesh.ufl_cell(), 1)
    FSdv = FunctionSpace(fluid_mesh, dve)


    if MPI.rank(MPI.comm_world) == 0:
        print("Define functions")

    # Create higher-order function on unrefined mesh for post-processing calculations
    u = Function(FSdv)
    
    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    flow_rate_output = []

    area_inlet = assemble(Constant(1.0) * dsi) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
    area_outlet3 = assemble(Constant(1.0) * dso3) # Get error: ufl.log.UFLException: This integral is missing an integration domain.

    print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(2, area_inlet))
    flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(2, area_inlet))

    print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(3, area_outlet3))
    flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(3, area_outlet3))


    # Get range of inlet and outlet ids in model
    inlet_outlet_min = 2 # lower bound for inlet and outlet IDs (inlet is usually 2)
    inlet_outlet_max = 9 # upper bound for inlet and outlet IDs

    bd_ids = np.unique(boundaries.array()[:])
    inlet_outlet_ids = bd_ids[(bd_ids >= inlet_outlet_min) & (bd_ids <=inlet_outlet_max)]

    if len(inlet_outlet_ids) > 2:
        dso4 = ds(4, domain=whole_mesh, subdomain_data=boundaries)
        area_outlet4 = assemble(Constant(1.0, name="one") * dso4) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
        print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(4,area_outlet4))
        flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(4,area_outlet4))

    if len(inlet_outlet_ids) > 3:
        dso5 = ds(5, domain=whole_mesh, subdomain_data=boundaries)
        area_outlet5 = assemble(Constant(1.0, name="one") * dso5) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
        print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(5,area_outlet5))
        flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(5,area_outlet5))   

    if len(inlet_outlet_ids) > 4:
        dso6 = ds(6, domain=whole_mesh, subdomain_data=boundaries)
        area_outlet6 = assemble(Constant(1.0, name="one") * dso6) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
        print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(6,area_outlet6))
        flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(6,area_outlet6))    

    # read velocity solution
    file_u = HDF5File(MPI.comm_world, file_path_u, "r")
    vec_name = "/velocity/vector_0"
    t_0 = file_u.attributes(vec_name)["timestamp"]
    vec_name = "/velocity/vector_1"
    t_1 = file_u.attributes(vec_name)["timestamp"]
    time_between_files = t_1 - t_0
    
    if MPI.rank(MPI.comm_world) == 0:
        print("Time between files: {}".format(time_between_files))
    
      # Start file counter
    file_counter = 0 
    n = FacetNormal(fluid_mesh)
    
    while True:

        try:
            # Read in solution to vector function
            file_u = HDF5File(MPI.comm_world, file_path_u, "r")
            u_name = "/velocity/vector_%d" % file_counter
            timestamp = file_u.attributes(u_name)["timestamp"]
            print("=" * 10, "Timestep: {}".format(timestamp), "=" * 10)
            file_u.read(u, u_name)
        except:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break   

        # Compute flow rate(s)
        flow_rate_inlet = assemble(inner(u, n)*dsi)
        flow_rate_outlet3 = assemble(inner(u, n)*dso3)
           
        flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(2,flow_rate_inlet))
        flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(3,flow_rate_outlet3))

        if MPI.rank(MPI.comm_world) == 0:
            print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(2,flow_rate_inlet))
            print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(3,flow_rate_outlet3))

        if 4 in inlet_outlet_ids:
            flow_rate_outlet4 = assemble(inner(u, n)*dso4)
            flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(4,flow_rate_outlet4))
            if MPI.rank(MPI.comm_world) == 0:
                print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(4,flow_rate_outlet4))

        if 5 in inlet_outlet_ids:
            flow_rate_outlet5 = assemble(inner(u, n)*dso5)
            flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(5,flow_rate_outlet5))
            if MPI.rank(MPI.comm_world) == 0:
                print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(5,flow_rate_outlet5))

        if 6 in inlet_outlet_ids:
            flow_rate_outlet6 = assemble(inner(u, n)*dso6)
            flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(6,flow_rate_outlet6))
            if MPI.rank(MPI.comm_world) == 0:
                print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(6,flow_rate_outlet6))

        file_counter += 1

    # Write flow rates to file
    flow_rate_file = open(file_path_flow_rate,"w")
    flow_rate_file.writelines(flow_rate_output)
    flow_rate_file.close()

if __name__ == '__main__':
    folder, mesh, _, _, stride, save_deg = read_command_line()
    compute_flow_rate(folder, mesh, stride, save_deg)
