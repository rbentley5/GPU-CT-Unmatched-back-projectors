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
# profiler.enable()


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

# Calls to AB- and BA-GMRES
X_lcurve_1, R_lcurve_1, lambdah_1 = hybrid_AB_GMRES(A,B,b,iter,ct.m, ct.n, ct.num_angles, p = p, regparam= 'lcurve_1', delta = delta)
X_lcurve_2, R_lcurve_2, lambdah_2 = hybrid_AB_GMRES(A,B,b,iter, ct.m, ct.n, ct.num_angles, p = p, regparam = 'none', delta = delta)


# ===============================================================================================
#  Plotting of results - Example has ended
# ===============================================================================================
# Computing the relative error between the solutions x_i and the true solution
res_lcurve_1 = np.zeros((iter,1))
res_lcurve_2 = np.zeros((iter,1))

for i in range(0,iter):
    res_lcurve_1[i] = np.linalg.norm(X.reshape(-1) - X_lcurve_1[:,i])/np.linalg.norm(X.reshape(-1))
    res_lcurve_2[i] = np.linalg.norm(X.reshape(-1) - X_lcurve_2[:,i])/np.linalg.norm(X.reshape(-1))

val_lcurve_1 = np.min(res_lcurve_1)
val_lcurve_2 = np.min(res_lcurve_2)

idx_lcurve_1 = np.argmin(res_lcurve_1)
idx_lcurve_2 = np.argmin(res_lcurve_2)


# Plotting
plt.figure()
plt.plot(range(0,iter),res_lcurve_1,'r-')
plt.plot(range(0,iter),res_lcurve_2,'m--')
plt.plot(idx_lcurve_1,val_lcurve_1,'r*')
plt.plot(idx_lcurve_2,val_lcurve_2,'m*')
plt.title('Hybrid AB GMRES L-curve comparison',fontname='cmr10',fontsize=16)
plt.xlabel('Iteration',fontname='cmr10',fontsize=16)
plt.ylabel('Relative error',fontname='cmr10',fontsize=16)
plt.legend(['Option 1', 'Option 2',
            'iter ='+str(idx_lcurve_1)+', error = '+str(round(val_lcurve_1,4)),
            'iter ='+str(idx_lcurve_2)+', error = '+str(round(val_lcurve_2,4))])
plt.savefig('test_images/testABBA.jpg', format="jpg", bbox_inches="tight")

plt.figure()
plt.plot(range(0, iter), lambdah_1, 'r-')
plt.plot(range(0, iter), lambdah_2, 'm-')
plt.legend(['Option 1', 'Option 2'])
plt.savefig('test_images/l_curve_lambdas'+ name+'_'+str(ct.num_pixels) + '_p'+str(p)+'_'+str(ct.num_angles)+'ang.jpg', format="jpg", bbox_inches="tight")

# xmin = np.min((np.min(ex1.X,),np.min(X_dp[:,10]),np.min(X_dp[:,20]),np.min(X_dp[:,50]),np.min(X_dp[:,100])))
# xmax = np.max((np.max(ex1.X,),np.max(X_dp[:,10]),np.max(X_dp[:,20]),np.max(X_dp[:,50]),np.max(X_dp[:,100])))
# xmin = np.min(ex1.X,)
# xmax = np.max(ex1.X,)

# num_pixels = ex1.num_pixels
# fig, axs = plt.subplots(1,5, figsize=(16,4))
# im0 = axs[0].imshow(X_dp[:,10].reshape(num_pixels, num_pixels),vmin=xmin,vmax=xmax)
# axs[0].set_title("AB-GMRES \n 10 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im0, ax=axs[0])
# im1 = axs[1].imshow(X_dp[:,20].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
# axs[1].set_title("AB-GMRES\n 20 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im1, ax=axs[1])
# im2 = axs[2].imshow(X_dp[:,50].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
# axs[2].set_title("AB-GMRES\n 50 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im2, ax=axs[2])
# im3 = axs[3].imshow(X_dp[:,100].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
# axs[3].set_title("AB-GMRES \n 100 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im2, ax=axs[3])
# im0 = axs[4].imshow(ex1.X,vmin=xmin,vmax=xmax)
# axs[4].set_title("Exact Image",fontname='cmr10',fontsize=16)
# plt.colorbar(im0, ax=axs[4])
# plt.savefig("ABreconstructions.jpg", format="jpg", bbox_inches="tight")

# fig, axs = plt.subplots(1,5, figsize=(16,4))
# im0 = axs[0].imshow(X_lcurve[:,10].reshape(num_pixels, num_pixels),vmin=xmin,vmax=xmax)
# axs[0].set_title("Hybrid AB-GMRES \n 10 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im0, ax=axs[0])
# im1 = axs[1].imshow(X_lcurve[:,20].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
# axs[1].set_title("Hybrid AB-GMRES\n 20 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im1, ax=axs[1])
# im2 = axs[2].imshow(X_lcurve[:,50].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
# axs[2].set_title("Hybrid AB-GMRES\n 50 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im2, ax=axs[2])
# im3 = axs[3].imshow(X_lcurve[:,100].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
# axs[3].set_title("Hybrid AB-GMRES \n 100 Iterations",fontname='cmr10',fontsize=16)
# plt.colorbar(im2, ax=axs[3])
# im0 = axs[4].imshow(ex1.X,vmin=xmin,vmax=xmax)
# axs[4].set_title("Exact Image",fontname='cmr10',fontsize=16)
# plt.colorbar(im0, ax=axs[4])


# plt.savefig("HyABreconstructions.jpg", format="jpg", bbox_inches="tight")
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("cumtime")
# stats.print_stats(20)  # top 20 slowest