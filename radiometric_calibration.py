import numpy as np
import utilities.create_inc as create_inc
import utilities.read_image as read_image
import config
import os
import time
import tifffile
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def radar_cross_section(length, wavelength, theta, phi):
  theta = np.deg2rad(theta)
  phi = np.deg2rad(phi)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta) + np.sin(theta) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs


def point_target_analysis(length, wavelength, elevation_angle, theta, azimuth, HH, VV):
  # correlator gain
  measured_rcs = np.abs(HH)
  theoritical_rcs = radar_cross_section(length = length, wavelength = wavelength, theta = theta + elevation_angle , phi = azimuth)

  a = (measured_rcs / theoritical_rcs)

  # co channel imbalance amplitude
  f = ((np.abs(VV)**2) / (np.abs(HH)**2)) ** 0.25

  # co channel imbalance phase
  co_phase = np.angle(np.array(VV) * np.conj(HH))

  return a, f, co_phase, measured_rcs, theoritical_rcs


def phase(phase_sum, phase_diff):
   phi_t = (phase_sum + phase_diff) / 2
   phi_r = (phase_sum - phase_diff) / 2
   return phi_t, phi_r
   

def distributed_target_analysis(HV, VH, gpu=False):
    # Cross-channel imbalance amplitude
    g = np.mean((np.abs(HV) ** 2) / (np.abs(VH))) ** 0.25

    # Cross-channel imbalance phase
    cross_phase = np.angle((HV * np.conjugate(VH)).mean())

    if gpu:
        
        import torch

        HV_tensor = torch.tensor(HV, device='cuda')
        VH_tensor = torch.tensor(VH, device='cuda')

        g_gpu = torch.mean((torch.abs(HV_tensor) ** 2) / torch.abs(VH_tensor)) ** 0.25
        cross_phase_gpu = torch.angle(torch.mean(HV_tensor * torch.conj(VH_tensor)))
        torch.cuda.empty_cache()
        return g_gpu.cpu().numpy(), cross_phase_gpu.cpu().numpy()

    return g, cross_phase


def apply_radio_cal(cr_length, input_csv_path, HH_array, HV_array, VH_array, VV_array):
  start = time.time()
  rows, cols = HH_array.shape
  
  with tqdm(total=rows*cols, desc='Calibration Progress', unit = ' pixels') as pbar:
    df = pd.read_csv(input_csv_path).copy()
    df['HH'] = df['HH'].apply(lambda x: np.complex64(complex(x)))
    df['VV'] = df['VV'].apply(lambda x: np.complex64(complex(x)))
    df['a'], df['f'], df['phase sum'], df['Measured RCS'], df['Theoritical RCS']  = point_target_analysis(length = df['Side Length'],
                                                                                                              wavelength = 0.234, 
                                                                                                              elevation_angle = df['Elevation Angle'], 
                                                                                                              theta = df['Incidence Angle'], 
                                                                                                              azimuth = df['Azimuth'], 
                                                                                                              HH = df['HH'], 
                                                                                                              VV = df['VV'])
    
    df = df[(df['Side Length'] == cr_length) & (df['Measured RCS'] > 10)]

    # Look up table
    look_up_dict = {}
    for _, row in df.iterrows():
      look_up_dict[row['Local Y'], row['Local X']] = {'A': row['a'], 'f': row['f'], 'P': row['phase sum']}


    g = (0.0069951843 / 0.0062046014) ** 0.25
    phase_diff = 0.279913

    inc = create_inc.create_inc_array_flat(min_look_angle = config.mininum_look_angle,
                                           max_look_angle = config.maximum_look_angle, 
                                           row_1x1 = config.row_1x1, 
                                           col_1x1 = config.col_1x1,
                                           subset = True)
    
    a_slope, a_intercept = np.polyfit(x = df['Incidence Angle'] - config.mount_antenna_angle, y = df['a'], deg = 1)
    p_slope, p_intercept = np.polyfit(x = df['Incidence Angle'], y = df['phase sum'], deg = 1)
    
    # Calibrated Image
    HH_cal = np.zeros_like(HH_array)
    HV_cal = np.zeros_like(HH_array)
    VH_cal = np.zeros_like(HH_array)
    VV_cal = np.zeros_like(HH_array)
    
    # Calibration Parameters
    A = np.zeros_like(HH_array)
    P = np.zeros_like(HH_array)

    for i in range(rows):
        for j in range(cols):

          if (i, j) in look_up_dict:
            A[i, j] = look_up_dict[(i, j)]["A"]
            P[i, j] = look_up_dict[(i, j)]["P"]
            f = look_up_dict[(i, j)]["f"]

          else:
              A[i, j] = a_slope * (inc[i, j] - config.mount_antenna_angle) + a_intercept
              P[i, j] = p_slope * (inc[i, j]) + p_intercept
              f = df['f'].mean()

          phi_t, phi_r = phase(phase_sum=P[i, j], phase_diff=phase_diff)
          HH_cal[i, j] = HH_array[i, j] / A[i, j]
          HV_cal[i, j] = HV_array[i, j] / (f * g * np.exp(1j * phi_t) * A[i, j])
          VH_cal[i, j] = VH_array[i, j] / ((f / g) * np.exp(1j * phi_r) * A[i, j])
          VV_cal[i, j] = VV_array[i, j] / (f * f * np.exp(1j * P[i, j]) * A[i, j])
          
          pbar.update(1)


  file_path = os.path.join(os.getenv('output_folder_path'),'Output.csv')
  df.to_csv(file_path, index = False)
  end = time.time()
  print(f'Executed in {(end - start)/60:.2f} minutes.')

  return HH_cal, HV_cal, VH_cal, VV_cal
        



if __name__ == '__main__':

  dataframe = os.getenv('Input_csv_path')
  HH_array = read_image.image_array(os.getenv('HH_tif_path'))
  HV_array = read_image.image_array(os.getenv('HV_tif_path'))
  VH_array = read_image.image_array(os.getenv('VH_tif_path'))
  VV_array = read_image.image_array(os.getenv('VV_tif_path'))

  HH_cal, HV_cal, VH_cal, VV_cal = apply_radio_cal(cr_length = 4.8, input_csv_path = dataframe, HH_array = HH_array, HV_array = HV_array, VH_array = VH_array, VV_array = VV_array)

  cal_files = [HH_cal, HV_cal, VH_cal, VV_cal]
  cal_file_names = ['HH_calibrated.tiff', 'HV_calibrated.tiff', 'VH_calibrated.tiff', 'VV_calibrated.tiff']

  uncal_files = [HH_array, HV_array, VH_array, VV_array]
  uncal_file_names = ['HH_uncalibrated.tiff', 'HV_uncalibrated.tiff', 'VH_uncalibrated.tiff', 'VV_uncalibrated.tiff']

  for i in range(4):
    path = os.getenv('output_folder_path')
    tifffile.imwrite(os.path.join(path, cal_file_names[i]), cal_files[i])
    tifffile.imwrite(os.path.join(path, uncal_file_names[i]), uncal_files[i])


  # normalize
  image = np.abs(HH_cal)
  norm = (image - np.min(image)) / (np.max(image) - np.min(image))
  plt.imshow(norm, cmap='gray',vmin=0.0001, vmax=0.0025)
  plt.show()
  
   


