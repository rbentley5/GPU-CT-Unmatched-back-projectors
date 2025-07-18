import astra

class fp_astra:
    '''ASTRA forward projector
    Forward projection models: 'line', 'strip', or 'linear' (Joseph)
    '''
    def __init__(self,ct_astra):
        self.num_pixels = ct_astra.num_pixels
        self.num_angles = ct_astra.num_angles
        self.num_dets   = ct_astra.num_dets 
        self.proj_id    = ct_astra.proj_id
        self.proj_geom  = ct_astra.proj_geom
        self.GPU        = ct_astra.GPU
        self.vol_geom   = ct_astra.vol_geom

    def apply_A(self,x):
        volume_id = astra.data2d.create('-vol', self.vol_geom, x.reshape(self.num_pixels,self.num_pixels))
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)
        if self.GPU == True:
            cfg = astra.creators.astra_dict('FP_CUDA')
        else:
            cfg = astra.creators.astra_dict('FP')
        cfg['ProjectorId'] = self.proj_id       # Id to forward projector A
        cfg['ProjectionDataId'] = sinogram_id   # Id to sinogram b (what A should be multiplied with)
        cfg['VolumeDataId'] = volume_id         # Id to size/volume of output
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra.algorithm.delete(alg_id)
        sinogram = astra.data2d.get(sinogram_id)

        return sinogram.reshape(-1)

    def __matmul__(self,x):
        return self.apply_A(x)

class bp_astra:
    '''ASTRA back projector'''
    def __init__(self,ct_astra):
        self.num_angles  = ct_astra.num_angles
        self.num_dets    = ct_astra.num_dets 
        self.vol_geom    = ct_astra.vol_geom
        self.proj_id     = ct_astra.proj_id
        self.proj_geom   = ct_astra.proj_geom
        self.GPU         = ct_astra.GPU
        self.sinogram_id = ct_astra.sinogram_id

    def apply_B(self,b):
        if self.GPU == True:
            sinogram_id = astra.data2d.create('-sino', self.proj_geom, b.reshape(self.num_angles,self.num_dets))
            recon_id = astra.data2d.create('-vol', self.vol_geom, 0)
            cfg = astra.creators.astra_dict('BP_CUDA')
            cfg['ProjectorId'] = self.proj_id
            cfg['ProjectionDataId'] = sinogram_id
            cfg['ReconstructionDataId'] = recon_id
            bp_id = astra.algorithm.create(cfg)
            astra.algorithm.run(bp_id)
            Bb = astra.data2d.get(recon_id)
        else:
            _, Bb = astra.create_backprojection(b.reshape(self.num_angles,self.num_dets), self.proj_id)
        return Bb.reshape(-1)

    def __matmul__(self,b):
        return self.apply_B(b)
