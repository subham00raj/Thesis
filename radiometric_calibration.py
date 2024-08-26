import numpy as np
import create_inc

def radar_cross_section(length, wavelength, elevation_angle, theta, azimuth):
  theta_cr = np.deg2rad(theta + elevation_angle)
  phi = np.deg2rad(azimuth)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta_cr) + np.sin(theta_cr) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs


def point_target_analysis(length, wavelength, elevation_angle, theta, azimuth, HH, VV):

  # correlator gain
  measured_rcs = np.abs(HH)
  theoritical_rcs = radar_cross_section(length, wavelength, elevation_angle, theta, azimuth)
  a = (theoritical_rcs / measured_rcs) ** 0.5

  # co channel imbalance amplitude
  f = ((np.abs(VV)**2) / (np.abs(HH))) ** 0.25

  # co channel imbalance phase
  phase_sum = np.angle(np.array(VV) * np.conj(HH))

  return a, f, phase_sum


def distributed_target_analysis(HV, VH, gpu=False):
    # Cross-channel imbalance amplitude
    g = np.mean((np.abs(HV) ** 2) / (np.abs(VH))) ** 0.25

    # Cross-channel imbalance phase
    phase_diff = np.angle(np.mean(HV * np.conj(VH)))

    if gpu:
        import torch

        HV_tensor = torch.tensor(HV, device='cuda')
        VH_tensor = torch.tensor(VH, device='cuda')

        g_gpu = torch.mean((torch.abs(HV_tensor) ** 2) / torch.abs(VH_tensor)) ** 0.25
        phase_diff_gpu = torch.angle(torch.mean(HV_tensor * torch.conj(VH_tensor)))

        return g_gpu.cpu().numpy(), phase_diff_gpu.cpu().numpy()

    return g, phase_diff


def apply_radio_cal(parameters):
   inc = create_inc.create_inc_array_flat(min_look_angle = 21.32159622, max_look_angle = 66.17122143, row_1x1 = 61349, col_1x1 = 9874)
   inc = inc[2000:5000, 41000:44000]
   
   pass



   