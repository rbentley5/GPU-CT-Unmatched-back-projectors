from solvers import *
from ct import ct_astra
from projectors import *
import csv
import numpy as np
import matplotlib.pyplot as plt
from trips.test_problems.Tomography import *

from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon
import cProfile, pstats

profiler = cProfile.Profile()


# reader = csv.reader(open('X_128_shepplogan_rnl03.csv'), delimiter=",")
# X = np.array(list(reader)).astype("float32")

# Tomo = Tomography()
# (x_true, nx, ny) = Tomo.gen_true('smooth', nx =1024, ny =1024)
# X = x_true.reshape(nx, ny).astype('float32')
img_size = 1024

base = shepp_logan_phantom()
X = resize(base, (img_size, img_size), order=1, mode='reflect', anti_aliasing=True).astype(np.float32)


# CT setup
num_pixels      = np.shape(X)[0]    # Number of pixels in x/y direction (X needs to be square)
num_dets        = num_pixels        # Number of detector pixels
num_angles      = 1800               # Number of projection angles
ang_start       = 0                 # Start angle in degress
ang_end         = 360               # End angle in degress
angles          = np.linspace(ang_start,ang_end,num_angles,dtype=int) / 180 * np.pi 
# %% ************************************ ASTRA ************************************
# Parameters for ASTRA
proj_model      = 'linear'          # The projection model: 'line', 'strip', or 'linear'
proj_geom       = 'parallel'         # The projection geometry: 'parallel' or 'fanflat'
source_origin   = 1000              # Distance from source to origin/center
origin_det      = 0                 # Distance from origin/center to detector
det_width       = 1                 # Detector width
gpu             = True              # Construct unmatched normal equations: 'False' or 'True' 
ct = ct_astra(num_pixels,num_angles,num_dets,angles,proj_model,proj_geom,source_origin,origin_det,det_width,gpu)

# Create Sinogram
_, Bexact = astra.create_sino(X, ct.proj_id)

# Create noisy sinogram
rnl     = 0.01
e0      = np.random.normal(0.0, 1.0, ct.m)
e1      = e0/np.linalg.norm(e0)
bexact  = Bexact.reshape(-1)
e       = rnl*np.linalg.norm(bexact)*e1
b       = bexact + e



A       = fp_astra(ct)     # The forward projector
B       = bp_astra(ct)     # The back projector  

iter    = 100               # Maximum number of iterations

name = 'head'
p = 10

# Reference to CT setup

# Create noisy sinogram
rnl     = 0.03
e0      = np.random.normal(0.0, 1.0, ct.m)
e1      = e0/np.linalg.norm(e0)
bexact  = Bexact.reshape(-1)
e       = rnl*np.linalg.norm(bexact)*e1
b       = bexact + e

delta = np.std(e)
print('delta = '+str(delta))
# Setup for ABBA methods
A       = fp_astra(ct)     # The forward projector
B       = bp_astra(ct)     # The back projector

iter    = 100               # Maximum number of iterations
# profiler.enable()

# Calls to AB- and BA-GMRES

user_input = input("Enter to start")
X_lcurve_1, R_lcurve_1= hybrid_AB_GMRES(A,B,b,iter,ct.m, ct.n, ct.num_angles, p = p, regparam= 'lcurve_2', delta = delta)


# X_lcurve_2, R_lcurve_2= hybrid_AB_GMRES(A,B,b,iter, ct.m, ct.n, ct.num_angles, p = p, regparam = 'lcurve_2', delta = delta)


# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("cumtime")
# stats.print_stats(20)  # top 20 slowest