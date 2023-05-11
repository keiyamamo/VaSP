'''
Based on the formatted data, compute the TKE and save it in a new file.
'''
import matplotlib.pyplot as plt
from postprocessing_common_h5py import read_npz_files
import argparse
import glob

def command_line_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute TKE from formatted data.')
    parser.add_argument('--df',  type=str, help='Folder containing the formatted data.')
    args = parser.parse_args()

    return args

def plot_tke(tke, v_bar, v_prime):
    some_point_id = 1
    plt.plot(v_bar.iloc[some_point_id], label='v_bar')
    plt.plot(v_prime.iloc[some_point_id], label='v_prime')
    plt.plot(tke.iloc[some_point_id], label='tke')
    plt.legend()
    plt.savefig('tke.png')
    # plt.show()
    
if __name__ == '__main__':
    args = command_line_parser()
    print("Reading npz files from " + args.df + " ...")
    tke = read_npz_files(glob.glob(args.df + '/tke.npz')[0])
    vx_bar = read_npz_files(glob.glob(args.df + '/vx_bar.npz')[0])
    vx_prime = read_npz_files(glob.glob(args.df + '/vx_prime.npz')[0])

    plot_tke(tke, vx_bar, vx_prime)