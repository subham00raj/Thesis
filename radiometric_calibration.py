import numpy as np

def radar_cross_section(length, wavelength, elevation_angle, theta, azimuth):
  theta_cr = np.deg2rad(theta + elevation_angle)
  phi = np.deg2rad(azimuth)
  factor = (4 * np.pi * (length**4)) / wavelength**2
  a = np.cos(theta_cr) + np.sin(theta_cr) * (np.cos(phi) + np.sin(phi))
  b = 2/a
  rcs = factor * ((a - b)**2)
  return rcs


def get_point_target_analysis_parameters(length, wavelength, elevation_angle, theta, azimuth, HH, HV, VH, VV):

  # correlator gain
  measured_rcs = np.abs(HH)
  theoritical_rcs = radar_cross_section(length, wavelength, elevation_angle, theta, azimuth)
  a = (theoritical_rcs / measured_rcs) ** 0.5

  # co-pol channel imbalance amplitude
  f = ((np.abs(VV)**2) / (np.abs(HH))) ** 0.25

  # phase calibration
  phase_sum = None

  return a, f, phase_sum