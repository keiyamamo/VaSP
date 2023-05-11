'''
Based on the formatted data, compute the TKE and save it in a new file.
'''

import numpy as np
from spectrograms import filter_time_data, get_sampling_constants
from postprocessing_common_h5py import read_npz_files
import argparse
import glob

def command_line_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute TKE from formatted data.')
    parser.add_argument('--df',  type=str, help='Folder containing the formatted data.')
    parser.add_argument('--start_t', type=float, default=0.0, help="Start time of simulation (s)")
    parser.add_argument('--end_t', type=float, default=0.05, help="End time of simulation (s)")
    
    args = parser.parse_args()

    return args

def compute_tke(vx, vy, vz, fs):
    """
    Given the vcelocity file for x, y, and z, compute the TKE and save it in a new file.
    Mathematically, TKE = 1/2 * (v_x_prime^2 + v_y_prime^2 + v_z_prime^2) where
    we assume that v = v_bar + v_prime, where v_bar is the mean velocity and v_prime is the fluctuating velocity.
    Args:

    Returns:

    """
    # Filter the data to create smoothed data
    vx_bar = filter_time_data(vx, fs)
    vy_bar = filter_time_data(vy, fs)
    vz_bar = filter_time_data(vz, fs)

    # Compute the fluctuating velocity
    vx_prime = vx - vx_bar
    vy_prime = vy - vy_bar
    vz_prime = vz - vz_bar
    from IPython import embed; embed(); exit(1)
    # Compute the TKE
    tke = 0.5 * (vx_prime**2 + vy_prime**2 + vz_prime**2)

    return vx_bar, vy_bar, vz_bar, vx_prime, vy_prime, vz_prime, tke

    
if __name__ == '__main__':
    args = command_line_parser()
    print("Reading npz files from " + args.df + " ...")
    vx = read_npz_files(glob.glob(args.df + '/*v_x.npz')[0])
    vy = read_npz_files(glob.glob(args.df + '/*v_y.npz')[0])
    vz = read_npz_files(glob.glob(args.df + '/*v_z.npz')[0])

    _, _, fs = get_sampling_constants(vx, args.start_t, args.end_t)
    print("Computing TKE ...")
    vx_bar, vy_bar, vz_bar, vx_prime, vy_prime, vz_prime, tke = compute_tke(vx, vy, vz, fs)
    # create dictionary to store variables
    variables = {'vx_bar': vx_bar,
                'vy_bar': vy_bar,
                'vz_bar': vz_bar,
                'vx_prime': vx_prime,
                'vy_prime': vy_prime,
                'vz_prime': vz_prime,
                'tke': tke}

    # save variables to file
    for var_name, var_value in variables.items():
        var_path = args.df + var_name + '.npz'
        print("Saving " + var_name + " to " + var_path + " ...")
        np.savez_compressed(var_path, component=var_value)
