from argparse import ArgumentParser
import re
from dolfin import *

parameters["allow_extrapolation"] = True


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="cyl_test", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--mesh', type=str, default="artery_coarse_rescaled", help="Mesh File Name",
                        metavar="PATH")
    parser.add_argument('--nu', type=float, default=3.5e-3, help="Fluid Viscosity used in simulation")
    parser.add_argument('--E_s', type=float, default=1e6, help="Elastic Modulus (Pascals) used in simulation")
    parser.add_argument('--nu_s', type=float, default=0.45, help="Poisson's Ratio used in simulation")
    parser.add_argument('--dt', type=float, default=0.001, help="Time step of simulation")
    parser.add_argument('--stride', type=int, default=1, help="Save frequency of simulation")    
    parser.add_argument('--save_deg', type=int, default=2, help="Input save_deg of simulation, i.e whether the intermediate P2 nodes were saved. Entering save_deg = 1 when the simulation was run with save_deg = 2 will result in only the corner nodes being used in postprocessing",
                        metavar="PATH")
    parser.add_argument('--start_t', type=float, default=0.0, help="Desired start time for postprocessing")
    parser.add_argument('--end_t', type=float, default=100.0, help="Desired end time for postprocessing")
    args = parser.parse_args()

    return args.case, args.mesh, args.nu, args.E_s, args.nu_s, args.dt, args.stride, args.save_deg, args.start_t, args.end_t

def get_time_between_files(xdmf_file):
    # Get the time between output files from an xdmf file
    file1 = open(xdmf_file, 'r') 
    Lines = file1.readlines() 
    time_ts=[]
    
    # This loop goes through the xdmf output file and gets the time value (time_ts)
    for line in Lines: 
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            time_ts.append(time)

    time_between_files = (time_ts[2] - time_ts[1]) # Calculate the time between files from xdmf file (in s)
    t_0 = time_ts[0]
    return t_0, time_between_files

def epsilon(u):
    """
    Computes the strain-rate tensor
    Args:
        u (Function): Velocity field

    Returns:
        epsilon (Function): Strain rate tensor of u
    """

    return 0.5 * (grad(u) + grad(u).T)


def local_project(f, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    a_proj = inner(u, v)*ds
    b_proj = inner(f, v)*ds
    A = assemble(a_proj, keep_diagonal=True)
    A.ident_zeros()
    b = assemble(b_proj)
    # solver = LocalSolver(A, b)
    # solver.factorize()
    u = Function(V)
    # solver.solve_local_rhs(u)
    solve(A, u.vector(), b)
    # from IPython import embed; embed(); exit(1)
    return u


class STRESS:
    def __init__(self, u, nu, mesh, velocity_degree):
        boundary_mesh = BoundaryMesh(mesh, 'exterior')
        self.bmV = VectorFunctionSpace(boundary_mesh, 'DG', velocity_degree -1)
        self.V = VectorFunctionSpace(mesh, 'DG', velocity_degree -1)

        # Compute stress tensor
        sigma = (2 * nu * sym(grad(u)))
        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        self.Ft = F - (Fn * n)  # vector-valued

    def __call__(self):
        """
        Compute stress for given velocity field u

        Returns:
            Ftv_mb (Function): Shear stress
        """
        self.Ftv = local_project(self.Ft, self.V)

        return self.Ftv