import astra
import GPUtil
import numpy as np

class ct_astra:
    '''Setup for ASTRA toolbox'''
    def __init__(self, num_pixels, num_angles, num_dets, angles, proj_model, proj_geom, source_origin, origin_det, det_width, GPU = True):
        '''Class setup for a 2D X-ray CT problem in ASTRA

        ----- INPUT -----
        num_pixels:     Number of pixels in an num_pixels x num_pixels image
        num_angles:     Number of angles
        num_dets:       Number of detector elements
        angles:         Angles in radians
        proj_model:     Projection model for CPU version choose between 'line', 'strip', or 'linear' (Joseph) 
                            and for GPU we only have Joseph
        proj_geom:      Projection geometry 'parallel' or 'fanflat'
        GPU:            True (use GPU) or False (use CPU)
        source_origin:  Distance between the source and the center of rotation
        origin_det:     Distance between the center of rotation and the detector array
        det_width:      Detector width
        '''
        # Set fields
        self.num_pixels     = num_pixels
        self.num_angles     = num_angles
        self.num_dets       = num_dets
        self.proj_angles    = angles
        self.m              = num_angles*num_dets
        self.n              = num_pixels**2
        self.vol_geom       = astra.create_vol_geom(self.num_pixels,self.num_pixels)
        self.GPU            = GPU

        # Check if there is a GPU connected to the host
        if GPU == True:
            n_device, = np.shape(GPUtil.getAvailable())

            if n_device < 1:
                raise Exception("No GPUs available.")

        # SETUP PROJECTION GEOMETRY
        # Parallel beam geometry
        if proj_geom == 'parallel': 
            self.proj_geom = astra.create_proj_geom(proj_geom, det_width, num_dets, self.proj_angles)
            if GPU == True:
                self.proj_id   = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
            else:
                self.proj_id   = astra.create_projector(proj_model, self.proj_geom, self.vol_geom)

        # Fan beam geometry
        elif proj_geom == 'fanflat':
            if proj_model == 'linear' and GPU == False:
                raise Exception("Fan beam geometry using the CPU can only handle strip and line")

            self.proj_geom = astra.create_proj_geom(proj_geom,det_width,num_dets,self.proj_angles,source_origin,origin_det)
            if GPU == True:
                self.proj_id   = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
            else:
                self.proj_id   = astra.create_projector(proj_model+'_fanflat', self.proj_geom, self.vol_geom)
        else: 
            raise Exception("Projection geometry can only be parallel or fanflat.")
        
        self.sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)
        self.recon_id = astra.data2d.create('-vol', self.vol_geom, 0)
    
    def deallocate(self):
        astra.data2d.delete(self.proj_id)

    def __del__(self):
        self.deallocate()
