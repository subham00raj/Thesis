import numpy as np
from tqdm import tqdm
from numpy.linalg import LinAlgError
np.set_printoptions(linewidth=1000)

def get_covariance(hh, hv, vh, vv):
    C = np.zeros((4,4), dtype=np.complex64)
    C[0,0] = np.mean(hh * np.conj(hh))
    C[0,1] = np.mean(hh * np.conj(hv))
    C[0,2] = np.mean(hh * np.conj(vh))
    C[0,3] = np.mean(hh * np.conj(vv))

    C[1,0] = np.mean(hv * np.conj(hh))
    C[1,1] = np.mean(hv * np.conj(hv))
    C[1,2] = np.mean(hv * np.conj(vh))
    C[1,3] = np.mean(hv * np.conj(vv))

    C[2,0] = np.mean(vh * np.conj(hh))
    C[2,1] = np.mean(vh * np.conj(hv))
    C[2,2] = np.mean(vh * np.conj(vh))
    C[2,3] = np.mean(vh * np.conj(vv))

    C[3,0] = np.mean(vv * np.conj(hh))
    C[3,1] = np.mean(vv * np.conj(hv))
    C[3,2] = np.mean(vv * np.conj(vh))
    C[3,3] = np.mean(vv * np.conj(vv))

    return C

# calibration matrix
def compute_sigma_1_matrix(u, v, w, z, alpha):
    sqrt_alpha = np.sqrt(alpha)
    arr = np.array([[1, -w, -v, v*w],
                    [-u / sqrt_alpha, 1 / sqrt_alpha, (u*v) / sqrt_alpha, -v / sqrt_alpha],
                    [-z / sqrt_alpha, (w*z) / sqrt_alpha, sqrt_alpha, -w / sqrt_alpha],
                    [(u*z), -z, -u, 1]])

    multiplier = 1 / (((u*w) - 1) * ((v*z)-1))  
    return arr * multiplier

def compute_alpha(covariance):
    x = covariance[1,2] / np.abs(covariance[1,2])
    y = np.sqrt(np.abs(covariance[1,1] / covariance[2,2]))
    return x*y

def compute_A(covariance):
     return (covariance[1,0] + covariance[2,0]) / 2

def compute_B(covariance):
     return (covariance[1,3] + covariance[2,3]) / 2

def compute_X_matrix(covariance, A, B):
    x_matrix = np.array([
          [covariance[2,0] - A],
          [covariance[1,0] - A],
          [covariance[2,3] - B], 
          [covariance[1,3] - B]])
    
    return x_matrix

def compute_zeta_matrix(covariance):
    zeta_matrix = np.array([
          [0, 0, covariance[3,0], covariance[0,0]],
          [covariance[0,0], covariance[3,0], 0, 0],
          [0, 0, covariance[3,3], covariance[0,3]],
          [covariance[0,3], covariance[3,3], 0, 0]])
    
    return zeta_matrix

def compute_tau_matrix(covariance):
    tau_matrix = np.array([
          [0, covariance[2,2], covariance[2,1], 0],
          [0, covariance[1,2], covariance[1,1], 0],
          [covariance[2,2], 0, 0, covariance[2,1]],
          [covariance[1,2], 0, 0, covariance[1,1]]])
    
    return tau_matrix



def parameter_updates(x, zeta, tau):
    X_Matrix_Need = np.concatenate([np.real(x), np.imag(x)])

    Zeta_Tau_Matrix_Need = np.concatenate([np.concatenate([np.real(zeta + tau), -1 * np.imag(zeta - tau)], axis=1),
                                           np.concatenate([np.imag(zeta + tau), np.real(zeta - tau)], axis=1)])

    delta_Solve = np.linalg.inv(Zeta_Tau_Matrix_Need) @ X_Matrix_Need

    delta_u = delta_Solve[0] + 1j * delta_Solve[4]
    delta_v = delta_Solve[1] + 1j * delta_Solve[5]
    delta_w = delta_Solve[2] + 1j * delta_Solve[6]
    delta_z = delta_Solve[3] + 1j * delta_Solve[7]

    return delta_u, delta_v, delta_w, delta_z


def ainsworth_cal(covar):
    u = 0
    v = 0
    w = 0
    z = 0
    gamma = 100
    tolerance = 1e-8
    i_iter = 0
    alpha = compute_alpha(covar)

    while i_iter < 12:
        sigma_1_matrix_i = compute_sigma_1_matrix(u, v, w, z, alpha)

        covariance_i = sigma_1_matrix_i @ covar @ np.conj(sigma_1_matrix_i).T
        A = compute_A(covariance_i)
        B = compute_B(covariance_i)

        X = compute_X_matrix(covariance_i, A, B)
        zeta = compute_zeta_matrix(covariance_i)
        tau = compute_tau_matrix(covariance_i)
        delta_u, delta_v, delta_w, delta_z = parameter_updates(X, zeta, tau)
        alpha_i = compute_alpha(covariance_i)

        u = (u + delta_u / np.sqrt(alpha))[0]
        v = (v + delta_v / np.sqrt(alpha))[0]
        w = (w + delta_w * np.sqrt(alpha))[0]
        z = (z + delta_z * np.sqrt(alpha))[0]
        alpha = alpha * alpha_i

        gamma = (max(abs(u), abs(v), abs(w), abs(z)))
        i_iter = i_iter + 1
    
    return sigma_1_matrix_i

    


