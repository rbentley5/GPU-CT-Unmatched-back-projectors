import numpy as np


def lcurve(H, b, tol = 1e-5, max_iter = 50) :
    
    
    def curvature_function(lambdah):
        y = np.linalg.lstsq(
            np.vstack((H, np.sqrt(lambdah) * np.identity(H.shape[1], dtype='float32'))),
            np.vstack((b.reshape(-1, 1), np.zeros((H.shape[1], 1), dtype='float32'))),
            rcond=None
        )[0]
        epsilon = np.linalg.norm(y) ** 2
        rho = H @ y - b.reshape(-1, 1)
        z = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*np.identity(H.shape[1], dtype = 'float32'))),
            np.vstack(( rho.reshape(-1, 1), np.zeros((H.shape[1], 1), dtype = 'float32') )), rcond=None)[0]
        rho = np.linalg.norm(rho) ** 2
        epsilon_prime = 4 / lambdah * np.dot(y.reshape(-1), z.reshape(-1))
        curve = ((2 * epsilon* rho) * ((lambdah**2) * epsilon_prime * rho + 2 * lambdah * epsilon * rho + (lambdah**4) * epsilon * epsilon_prime )) / (epsilon_prime * ((lambdah**2) * (epsilon**2) + rho**2)**(3/2))
        return curve
    # Golden section search parameters
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    invphi = 1 / phi
    invphi2 = invphi**2

    # Search interval for lambda (adjust if needed)
    lambda_low = 1e-10 
    lambda_high = 1e3
    lambda_2 = lambda_high - invphi * (lambda_high - lambda_low)
    lambda_1 = lambda_low + invphi * (lambda_high - lambda_low)

    curve1 = curvature_function(lambda_1)
    curve2 = curvature_function(lambda_2)

    iter_count = 0

    while np.abs(lambda_high- lambda_low) > tol and iter_count < max_iter :
        iter_count += 1

        if curve1 < curve2:
            lambda_high = lambda_low
            lambda_2 = lambda_1
            curve2 = curve1
            lambda_1 = lambda_high - invphi*(lambda_high - lambda_low)
            curve1 = curvature_function(lambda_1)
        else:
            lambda_low = lambda_1
            lambda_1 = lambda_2
            curve1 = curve2
            lambda_2 = lambda_low + invphi*(lambda_high - lambda_low)
            curve2 = curvature_function(lambda_2)
    optimal_lambda = (lambda_low + lambda_high) / 2

    return optimal_lambda

def NCP(r, m, num_angles):
    ''' 
    Stopping criteria: Normalized Cumulative Periodogram

    INPUTS
    r:      Residual vector for i'th iteration.
    m:      Number of pixels in the sinogram.
    num_angles:  The number of view angles.
    '''
    
    nt = int(num_angles)
    nnp = int(m / nt)
    q = int(np.floor(nnp/2))
    c_white = np.linspace(1,q,q)/q
    C = np.zeros((q,nt))
    
    R = r.reshape(nnp,nt)
    for j in range(0,nt):
        RKH = np.fft.fft(R[:,j])
        pk = abs(RKH[0:q+1])
        c = np.cumsum(pk[1:])/np.sum(pk[1:])
        C[:,j] = c

    Nk = np.linalg.norm(np.mean(C,1)-c_white)
    return Nk

