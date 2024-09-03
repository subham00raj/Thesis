import numpy as np
import matplotlib.pyplot as plt

def polsign(rcs, mode='c'):
    dic = {'c': 'Co', 'x': 'Cross'}
    
    el_range = np.linspace(-45, 45, 200)
    tilt_range = np.linspace(-90, 90, 200)
    
    El, Tilt = np.meshgrid(el_range, tilt_range)
    
    # Compute the electric field vectors
    tau_rad = np.deg2rad(Tilt.flatten())
    epsilon_rad = np.deg2rad(El.flatten())
    
    Ei = np.array([
        np.cos(tau_rad) * np.cos(epsilon_rad) - 1j * np.sin(tau_rad) * np.sin(epsilon_rad),
        np.sin(tau_rad) * np.cos(epsilon_rad) + 1j * np.cos(tau_rad) * np.sin(epsilon_rad)
    ])
    
    # Compute the scattered electric field
    Es = np.dot(rcs, Ei)
    
    NumPts = len(tau_rad)
    pResp = np.zeros(NumPts, dtype=complex)
    
    if mode == 'c':
        Er = Ei
    else:
        Er = np.conj(np.flipud(Ei))
        Er[0, :] = -Er[0, :]
    
    # Calculate the polarization response
    for m in range(NumPts):
        pResp[m] = np.dot(Er[:, m].T, Es[:, m])
    
    pResp = np.abs(pResp)
    pResp = pResp / np.max(pResp)
    pResp = pResp.reshape(len(tilt_range), len(el_range))
    
    # Create a 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(El, Tilt, pResp, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Ellipticity Angle (degrees)', fontsize=12)
    ax.set_ylabel('Tilt (degrees)', fontsize=12)
    ax.set_zlabel('Normalized Polarization Signature', fontsize=12)
    ax.set_title(f'{dic.get(mode)}-Polarization Signature', fontsize=14)
    ax.view_init(elev=30, azim=-120)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rcs_matrix = np.array([[1, 0], [0, 1]])
    polsign(rcs_matrix, 'c')

