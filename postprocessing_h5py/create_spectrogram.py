import os
import numpy as np
import spectrograms as spec
import matplotlib.pyplot as plt

"""
This script creates spectrograms from formatted matrices (.npz files)"

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 
    lowcut: High pass filter cutoff frequency (Hz)
    ylim: y limit of spectrogram graph")
    r_sphere: Sphere in which to include points for spectrogram, this is the sphere radius
    x_sphere: Sphere in which to include points for spectrogram, this is the x coordinate of the center of the sphere (in m)
    y_sphere: Sphere in which to include points for spectrogram, this is the y coordinate of the center of the sphere (in m)
    z_sphere: Sphere in which to include points for spectrogram, this is the z coordinate of the center of the sphere (in m)
    dvp: "d", "v", "p", or "wss", parameter to postprocess
    interface_only: uses nodes at the interface only. Used for wall pressure spectrogram primarily

"""

def create_spectrogram_composite(case_name, dvp, df, start_t, end_t, 
                                 nWindow_per_sec, overlapFrac, 
                                 window, lowcut, thresh_val, max_plot, imageFolder, 
                                 flow_rate_file=None, amplitude_file=None,
                                 power_scaled=False):


    # Calculate number of windows (you can adjust this equation to fit your temporal/frequency resolution needs)
    nWindow = np.round(nWindow_per_sec*(end_t-start_t))+3

    # Get sampling constants
    T, _, fs = spec.get_sampling_constants(df,start_t,end_t)


    # High-pass filter dataframe for spectrogram
    df_filtered = spec.filter_time_data(df,fs,
                                        lowcut=lowcut,
                                        highcut=15000.0,
                                        order=6,
                                        btype='highpass')

    # Specs with Reyynolds number
    # IMPORTATNT: This is a hack to slice the dataframe to only include the second cardiac cycle. 
    #             The number 2600 is the index of the second cardiac cycle and depends on the simulation
    df_filtered = df_filtered.iloc[:,2600::]
    bins, freqs, Pxx, max_val, min_val, lower_thresh = spec.compute_average_spectrogram(df_filtered, 
                                                                                        fs, 
                                                                                        nWindow,
                                                                                        overlapFrac,
                                                                                        window,
                                                                                        start_t,
                                                                                        end_t,
                                                                                        thresh_val,
                                                                                        scaling="spectrum",
                                                                                        filter_data=False,
                                                                                        thresh_method="old")
    # If 
    # bins = bins+start_t # Need to shift bins so that spectrogram timing is correct
    
    # create separate spectrogram figure
    fig2, ax2_1 = plt.subplots()
    fig2.set_size_inches(7.5, 5) #fig1.set_size_inches(10, 7)
    title = "Pxx max = {:.2e}, Pxx min = {:.2e}, threshold Pxx = {}".format(max_val, min_val, lower_thresh)
    spec.plot_spectrogram(fig2,ax2_1,bins,freqs,Pxx,ylim,title=title,x_label="Time (s)",color_range=[thresh_val,max_plot])
    fullname = dvp+"_"+case_name + '_'+str(nWindow)+'_windows_'+'_'+"thresh"+str(thresh_val)+"_spectrogram"
    path_to_fig = os.path.join(imageFolder, fullname + '.png')
    fig2.savefig(path_to_fig)


if __name__ == '__main__':
    # Load in case-specific parameters
    case_path, mesh_name, save_deg, stride,  start_t, end_t, lowcut, ylim, r_sphere, x_sphere, y_sphere, z_sphere, dvp, _, _, interface_only, sampling_method, component, _, point_id = spec.read_command_line_spec()

    # Read fixed spectrogram parameters from config file
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Spectrogram.config")
    overlapFrac, window, n_samples, nWindow_per_sec, lowcut, thresh_val, max_plot, amplitude_file_name, flow_rate_file_name = spec.read_spec_config(config_file,dvp)

    # Create or read in spectrogram dataframe
    dvp, df, case_name, case_path, imageFolder, visualization_hi_pass_folder  = spec.read_spectrogram_data(case_path, 
                                                                                                           mesh_name, 
                                                                                                           save_deg, 
                                                                                                           stride, 
                                                                                                           start_t, 
                                                                                                           end_t, 
                                                                                                           n_samples, 
                                                                                                           ylim, 
                                                                                                           r_sphere, 
                                                                                                           x_sphere, 
                                                                                                           y_sphere, 
                                                                                                           z_sphere, 
                                                                                                           dvp, 
                                                                                                           interface_only, 
                                                                                                           component,
                                                                                                           point_id,
                                                                                                           flow_rate_file_name='MCA_10',
                                                                                                           sampling_method=sampling_method)
    

    # Create spectrograms
    create_spectrogram_composite(case_name, 
                                 dvp, 
                                 df,
                                 start_t, 
                                 end_t, 
                                 nWindow_per_sec, 
                                 overlapFrac, 
                                 window, 
                                 lowcut,
                                 thresh_val, 
                                 max_plot, 
                                 imageFolder, 
                                 flow_rate_file=None,
                                 amplitude_file=None,
                                 power_scaled=False)
