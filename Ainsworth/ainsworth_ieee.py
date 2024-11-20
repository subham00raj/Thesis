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

    # Initialize cross talk parameters with zero
    u = 0
    v = 0
    w = 0
    z = 0
    gamma = 100
    tolerance = 1e-8
    i_iter = 0
    alpha = compute_alpha(covar)

    while i_iter < 2:
        sigma_1_matrix_i = compute_sigma_1_matrix(u, v, w, z, alpha)

        covariance_i = sigma_1_matrix_i @ covar @ np.conj(sigma_1_matrix_i).T
        A = compute_A(covariance_i)
        B = compute_B(covariance_i)

        X = compute_X_matrix(covariance_i, A, B)
        zeta = compute_zeta_matrix(covariance_i)
        tau = compute_tau_matrix(covariance_i)
        delta_u, delta_v, delta_w, delta_z = parameter_updates(X, zeta, tau)
        alpha_i = compute_alpha(covariance_i)

        u = (u + delta_u / alpha)[0]
        v = (v + delta_v / alpha)[0]
        w = (w + delta_w / alpha)[0]
        z = (z + delta_z / alpha)[0]
        alpha = alpha * alpha_i

        gamma = (max(abs(u), abs(v), abs(w), abs(z)))
        print(f'This is gamma in {i_iter}th iteration :', gamma)
        i_iter = i_iter + 1
    
    return gamma

    
    
######################################################################### I M P L M E N T A T I O N ####################################################################################

if __name__ == '__main__':
    
    HH = np.array([[-0.00690127-0.02906244j,  0.04874293-0.01083722j,  0.09196864-0.0122088j ],
        [-0.01627338-0.00610797j,  0.01897904+0.04147513j,  0.06223828+0.06260392j],
        [-0.05928316-0.01689963j, -0.02648653+0.03150305j, -0.00076173+0.05830637j]], dtype=np.complex128)

    HV = np.array([[-0.00253268-0.00165899j, -0.00100378-0.00021214j, -0.00176983+0.00142684j],
        [-0.00669271-0.00030713j,  0.0008027 +0.00139327j,  0.00277857-0.00027674j],
        [-0.00810817+0.00418459j, -0.00066056+0.0070381j ,  0.00743048+0.0025053j ]], dtype=np.complex128)

    VH = np.array([[ 0.00194239+0.00601709j, -0.00533496+0.00506952j, -0.00875523+0.00340825j],
        [ 0.00257253+0.00342182j,  0.00220111-0.00319489j, -0.00045381-0.00870172j],
        [-0.00121244+0.00530288j,  0.0023198 +0.00225626j,  0.00704284-0.0050623j ]], dtype=np.complex128)

    VV = np.array([[ 0.00247177-0.03491862j,  0.06061775-0.03294461j,  0.10629158-0.04479036j],
        [-0.00996231+0.00387164j,  0.0372977 +0.02979462j,  0.08158179+0.03785375j],
        [-0.05961524-0.00431815j, -0.01763038+0.03711419j,  0.00664863+0.05308216j]], dtype=np.complex128)


    u_list = []
    v_list = []
    w_list = []
    z_list = []
    alpha_list = []
    window_size = 3

    def sliding_window(HH, HV, VH, VV, window_size):
        rows,cols = HH.shape
        for i in range(rows):
            for j in range(0, cols, window_size):
                hh = HH[i, j:j+window_size]
                hv = HV[i, j:j+window_size]
                vh = VH[i, j:j+window_size]
                vv = VV[i, j:j+window_size]

                yield hh, hv, vh, vv

    pbar = tqdm(total = (HH.shape[0] * HH.shape[1])/ window_size, desc = 'Calculating Parameters', unit = 'windows')
    for a,b,c,d in sliding_window(HH, HV, VH, VV, window_size):
        cov = get_covariance(a,b,c,d)
        ainsworth_cal(cov)
        pbar.update(1)
    pbar.close()


