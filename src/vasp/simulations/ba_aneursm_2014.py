"""
Problem file for BA aneurysm 2014 FSI simulation
"""
import numpy as np

from turtleFSI.problems import *
from dolfin import HDF5File, Mesh, MeshFunction, FacetNormal, \
    DirichletBC, Measure, inner, parameters, assemble

from vasp.simulations.simulation_common import load_probe_points, calculate_and_print_flow_properties, \
    print_probe_points, load_solid_probe_points, print_solid_probe_points

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):

    # Compute some solid parameters
    # Need to stay here since mus_s and lambda_s are functions of nu_s and E_s
    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    default_variables.update(dict(
        # Temporal parameters
        T=1.902,  # Simulation end time
        dt=0.0001902,  # Timne step size
        theta=0.501,  # Theta scheme parameter
        save_step=1,  # Save frequency of files for visualisation
        save_solution_after_tstep=951,  # Start saving the solution after this time step for the mean value
        checkpoint_step=50,  # Save frequency of checkpoint files
        # Linear solver parameters
        linear_solver="mumps",
        atol=1e-6,  # Absolute tolerance in the Newton solver
        rtol=1e-6,  # Relative tolerance in the Newton solver
        recompute=20,  # Recompute the Jacobian matix within time steps
        recompute_tstep=20,  # Recompute the Jacobian matix over time steps
        # boundary condition parameters
        id_in=[5, 4],  # Inlet boundary IDs corase mesh
        inlet_outlet_s_id=11,  # inlet and outlet id for solid
        fsi_id=22,  # id for fsi surface
        rigid_id=11,  # "rigid wall" id for the fluid
        outer_id=33,  # id for the outer surface of the solid
        # Fluid parameters
        Q_mean=1.25E-06,
        P_mean=11200,
        T_Cycle=0.951,  # Used to define length of flow waveform
        rho_f=1.000E3,  # Fluid density [kg/m3]
        mu_f=3.5E-3,  # Fluid dynamic viscosity [Pa.s]
        dx_f_id=1,  # ID of marker in the fluid domain
        # mesh lifting parameters (see turtleFSI for options)
        extrapolation="laplace",
        extrapolation_sub_type="constant",
        # Solid parameters
        rho_s=1.0E3,  # Solid density [kg/m3]
        mu_s=mu_s_val,  # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,  # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus [Pa]
        dx_s_id=2,  # ID of marker in the solid domain
        # Simulation parameters
        folder="ba_aneurysm_results_fsi",  # Folder name generated for the simulation
        mesh_path="mesh/file_aneurysm.h5",
        FC_file="FC_MCA_10",  # File name containing the fourier coefficients for the flow waveform
        P_FC_File="FC_Pressure",  # File name containing the fourier coefficients for the pressure waveform
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
        save_deg=2,  # Degree of the functions saved for visualisation
        scale_probe=True,  # Scale the probe points to meters
    ))

    return default_variables


def get_mesh_domain_and_boundaries(mesh_path, **namespace):

    # Read mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")
   
    return mesh, domains, boundaries


def create_bcs(NS_expressions, mesh, T, dt, nu, V, Q, id_in, id_out, vel_t_ramp, t, **NS_namespace):
    # Variables needed during the simulation
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    # ID for boundary conditions
    id_lva = id_in[0]
    id_rva = id_in[1]

    if MPI.rank(MPI.comm_world) == 0:
        print("LVA ID: ", id_lva)
        print("RVA ID: ", id_rva)
    
    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    # Fluid velocity BCs
    dsi1 = ds(id_lva)
    dsi2 = ds(id_rva)

    # inlet area
    inlet_area1 = assemble(1 * dsi1)
    inlet_area2 = assemble(1 * dsi2)

    if MPI.rank(MPI.comm_world) == 0:
        print("Inlet area1: ", inlet_area1)
        print("Inlet area2: ", inlet_area2)

    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni1 = np.array([assemble(n[i] * dsi1) for i in range(ndim)])
    ni2 = np.array([assemble(n[i] * dsi2) for i in range(ndim)])
    n_len1 = np.sqrt(sum([ni1[i] ** 2 for i in range(ndim)]))
    n_len2 = np.sqrt(sum([ni2[i] ** 2 for i in range(ndim)]))
    normal1 = ni1 / n_len1
    normal2 = ni2 / n_len2
    
    if MPI.rank(MPI.comm_world) == 0:
        print("Normal1: ", normal1)
        print("Normal2: ", normal2)
    
    lva_velocity = [0.848637616,
                    0.80701909,
                    0.847125648,
                    0.938298849,
                    1.254212349,
                    1.891392824,
                    2.027271038,
                    1.77177903,
                    1.792383614,
                    1.913592303,
                    1.888049839,
                    1.740999381,
                    1.520779612,
                    1.344482404,
                    1.382113934,
                    1.499291641,
                    1.443681085,
                    1.329292421,
                    1.29572188,
                    1.23181514,
                    1.178122082,
                    1.174780102,
                    1.136369058,
                    1.022520303,
                    0.93012271,
                    0.912917651,
                    0.920154996,
                    0.895582053,
                    0.942527649,
                    0.936963696]
        
    rva_velocity = [0.247426581,
                    0.422073122,
                    0.810264593,
                    1.229616588,
                    2.163382059,
                    3.7332934,
                    4.550075292,
                    4.245502431,
                    4.118670492,
                    3.095260393,
                    2.180491478,
                    1.521510309,
                    0.758527135,
                    0.368080821,
                    0.481735522,
                    0.380346012,
                    0.185040665,
                    0.10628233,
                    0.119750561,
                    0.234517943,
                    0.396352626,
                    0.375314551,
                    0.332350363,
                    0.251735247,
                    0.234669003,
                    0.182901337,
                    0.299768979,
                    0.433334731,
                    0.381273636,
                    0.319951809]
    
    t_values = np.linspace(0, 951, len(lva_velocity))
    spl = UnivariateSpline(t_values, lva_velocity, k=5)

    tnew = np.linspace(0, 951, 1000)
    spl.set_smoothing_factor(0.1)
    lva = spl(tnew)
    
    # Inflow at lva
    tmp_area, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, id_in[0], boundaries)
    inlet_lva = make_womersley_bcs(tnew, lva, nu, tmp_center, tmp_radius, tmp_normal, V.ufl_element())
    for uc in inlet_lva:
        uc.set_t(t)

    # Inflow at rva
    tmp_area, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, id_in[1], boundaries)
    interp_rva = UnivariateSpline(t_values, rva_velocity, k=5)
    interp_rva.set_smoothing_factor(0.1)
    rva = interp_rva(tnew)
    
    inlet_rva = make_womersley_bcs(tnew, rva, nu, tmp_center, tmp_radius, tmp_normal, V.ufl_element())
    for uc in inlet_rva:
        uc.set_t(t)

    u_inlet_lva = [DirichletBC(DVP.sub(1).sub(i), inlet_lva[i], boundaries, id_lva) for i in range(3)]
    u_inlet_rva = [DirichletBC(DVP.sub(1).sub(i), inlet_rva[i], boundaries, id_rva) for i in range(3)]
    
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)
    # Assemble boundary conditions
    bcs = u_inlet_lva + + u_inlet_rva+ [u_inlet_s]

    # Load Fourier coefficients for the pressure and scale by flow rate
    An_P, Bn_P = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), P_FC_File)).T

    # Apply pulsatile pressure at the fsi interface by modifying the variational form
    n = FacetNormal(mesh)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    p_out_bc_val = InnerP(t=0.0, t_start=0.2, t_ramp=0.4, An=An_P, Bn=Bn_P, period=T_Cycle, P_mean=P_mean, degree=p_deg)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+')) * dSS(fsi_id)
    
    return dict(bcs=bcs, inlet_lva=inlet_lva, inlet_rva=inlet_rva, p_out_bc_val=p_out_bc_val, F_solid_linear=F_solid_linear, n=n, dsi=dsi,
                inlet_area=inlet_area1)


def initiate(mesh_path, scale_probe, mesh, v_deg, p_deg, **namespace):

    probe_points = load_probe_points(mesh_path)
    # In case the probe points are in mm, scale them to meters
    if scale_probe:
        probe_points = probe_points * 0.001
    solid_probe_points = load_solid_probe_points(mesh_path)

    Vv = VectorFunctionSpace(mesh, "CG", v_deg)
    V = FunctionSpace(mesh, "CG", p_deg)
    d_mean = Function(Vv)
    u_mean = Function(Vv)
    p_mean = Function(V)

    return dict(probe_points=probe_points, d_mean=d_mean, u_mean=u_mean, p_mean=p_mean, solid_probe_points=solid_probe_points)


def pre_solve(t, inlet_lva, inlet_rva, p_out_bc_val, **namespace):
    for uc in inlet_lva:
        # Update the time variable used for the inlet boundary condition
        uc.set_t(t)

        # Multiply by cosine function to ramp up smoothly over time interval 0-250 ms
        if t < 0.25:
            uc.scale_value = -0.5 * np.cos(np.pi * t / 0.25) + 0.5
        else:
            uc.scale_value = 1.0

    for uc in inlet_rva:
        # Update the time variable used for the inlet boundary condition
        uc.set_t(t)

        # Multiply by cosine function to ramp up smoothly over time interval 0-250 ms
        if t < 0.25:
            uc.scale_value = -0.5 * np.cos(np.pi * t / 0.25) + 0.5
        else:
            uc.scale_value = 1.0

    # Update pressure condition
    p_out_bc_val.update(t)

    return dict(inlet_lva=inlet_lva, inlet_rva=inlet_rva, p_out_bc_val=p_out_bc_val)


def post_solve(dvp_, n, dsi, dt, mesh, inlet_area, mu_f, rho_f, probe_points, t,
               save_solution_after_tstep, d_mean, u_mean, p_mean, **namespace):
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    print_probe_points(v, p, probe_points)
    print_solid_probe_points(d, probe_points)
    calculate_and_print_flow_properties(dt, mesh, v, inlet_area, mu_f, rho_f, n, dsi)

    if t >= save_solution_after_tstep * dt:
        # Here, we accumulate the velocity filed in u_mean
        d_mean.vector().axpy(1, d.vector())
        u_mean.vector().axpy(1, v.vector())
        p_mean.vector().axpy(1, p.vector())
        return dict(u_mean=u_mean, d_mean=d_mean, p_mean=p_mean)
    else:
        return None


def finished(d_mean, u_mean, p_mean, visualization_folder, save_solution_after_tstep, T, dt, **namespace):
    # Divide the accumulated vectors by the number of time steps
    num_steps = T / dt - save_solution_after_tstep + 1
    for data in [d_mean, u_mean, p_mean]:
        data.vector()[:] = data.vector()[:] / num_steps

    # Save u_mean as a XDMF file using the checkpoint
    data_names = [
        (d_mean, "d_mean.xdmf"),
        (u_mean, "u_mean.xdmf"),
        (p_mean, "p_mean.xdmf")
    ]

    for vector, data_name in data_names:
        file_path = os.path.join(visualization_folder, data_name)
        with XDMFFile(MPI.comm_world, file_path) as f:
            f.write_checkpoint(vector, data_name, 0, XDMFFile.Encoding.HDF5)
