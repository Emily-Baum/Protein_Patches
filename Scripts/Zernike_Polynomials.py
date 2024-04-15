# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 03:08:46 2023

@author: Emily
"""

#%%

def zernike_polynomials(voxels,N=20):
    
    """

    Parameters
    ----------
    X : numpy array of float
    
        The coordinates of the points at which to calculate the value of the 
        3D Zernike function.
        
    n : int, optional
    
        Order of approximation of the 3D Zernike function. Higher order provides
        higher resolution but scales computation exponentially. The default is 20.

    Returns
    -------
    Z : list of float
    
        Calculates the value of the 3D Zernike function at a given coordinate within a unit sphere.
        Equations numbers follow: https://doi.org/10.1002/prot.22030.

    """
    
    import math
    import numpy as np

    r = (voxels[:,0]**2 + voxels[:,1]**2 + voxels[:,2]**2)**.5

    Z = []
    for n in range(0,N+1):
        for l in range(n+1): # l goes from 0 to n
            if ((n-l)%2) == 0: # (n-l) even
                qsum = 0
                k = int((n-l)/2) # 2k = n - l
                
                for v in range(k+1): # v goes from 0 to k
                    
                    # eqn 8
                    qklv = (((-1)**k) / (2**(2*k)) * (((2*l+4*k+3)/3)**.5) * math.comb(2*k,k) 
                         * ((-1)**v) * ((math.comb(k,v) * math.comb(2*(k+l+v)+1,2*k)) / math.comb(k+l+v,k)))
                    
                    # sum portion of eqn 7
                    qklv *= r**(2*v) 
                    qsum += qklv
                    
                rl = r**l
                
                for m in range(-l,l+1): # m goes from -l to l
                    
                    # eqn 6
                    clm = (((2*l+1)*math.factorial(l+m)*math.factorial(l-m))**.5)/math.factorial(l) 
                    usum = 0
                    
                    # sum portion of eqn 5
                    for u in range(math.floor((l-m)/2)+1): # u goes from 0 to floor of (l-m)/2
                        if m+u < 0:
                            comb = 0
                        else:
                            comb = math.comb((l-u),(m+u))
                        usum += math.comb(l,u)*comb*((-(voxels[:,0]**2 + voxels[:,1]**2)/(4*voxels[:,2]**2))**u)
                    
                    # eqn 5
                    elm = rl*clm*(((1j*voxels[:,0]-voxels[:,1])/2)**m)*(voxels[:,2]**(l-m))*usum 
                    # eqn 7
                    Znlm = elm*qsum
                    Z.append(Znlm)
    Z = np.asarray(Z).T
    return Z
        
#%%

def voxelize(scaled_patch,gridpoints=20):
    """
    
    Parameters
    ----------
    scaled_patch : numpy array of float
    
        The dictionary contains the scaled coordinates of all surface points
        (and opt atom centers) within patch_radius of a random surface point.
        Dictionary entries are named by the centerpoint_index of each patch.
        
    gridpoints : int, optional
    
        The number of voxels in each dimension to construct a unit grid (gridpoints**3).
        Only the voxels within a unit sphere are retained. The default is 20.

    Returns
    -------
    voxels : list of float
    
        The coordinates of each voxel certerpoint.
        
    voxel_surf: numpy array of int
    
        The mapping of the surface patch to the voxel grid. The values in the 
        list are 1 if the voxel contains part of the surface, or 0 if it does not.

    """
    
    import numpy as np
    
    x = np.linspace(-1,1,gridpoints)
    voxels = [[i,j,k] for i in x for j in x for k in x if i**2 + j**2 + k**2 <= 1]
    
    voxel_surf = voxel_surface(scaled_patch,voxels,gridpoints)
    
    #TODO
    #voxel_feat = voxel_features(scaled_patch,features,voxels,gridpoints)
    
    return voxels, voxel_surf

#%%

def voxel_surface(scaled_patch,voxels,gridpoints):
    """
    
    Parameters
    ----------
    scaled_patch : dict of numpy arrays
    
        The dictionary contains the scaled coordinates of all surface points
        (and opt atom centers) within patch_radius of a random surface point.
        Dictionary entries are named by the centerpoint_index of each patch.
        
    voxels : list of float
    
        The coordinates of each voxel certerpoint.
        
    gridpoints : int
    
        The number of voxels in each dimension to construct a unit grid (gridpoints**3).
        Only the voxels within a unit sphere are retained. The default is 20.

    Returns
    -------
    voxel_surf : numpy array of int
    
        The mapping of the surface patch to the voxel grid. The values in the 
        list are 1 if the voxel contains part of the surface, or 0 if it does not.

    """

    import numpy as np    

    voxel_spacing = 1/(gridpoints-1)
    
    voxel_surf = np.zeros(len(voxels))
    for v_coords in voxels:
        for p_coords in scaled_patch:
            if abs(v_coords[0] - p_coords[0]) < voxel_spacing:
                if abs(v_coords[1] - p_coords[1]) < voxel_spacing:
                    if abs(v_coords[2] - p_coords[2]) < voxel_spacing:
                        voxel_surf[voxels.index(v_coords)] = 1
                        continue
    return voxel_surf

#%% 
# TODO work in progress
def zernike_descriptor(components,voxel_surf,gridpoints=20,N=20):
    
    import numpy as np
    
    zernike_fit = np.linalg.lstsq(components,voxel_surf,rcond=None)[0]
    
    voxel_spacing = 1/(gridpoints-1)
    
    zernike_moments = 3/(4*np.pi)*zernike_fit*voxel_spacing**3
    sq_moments = zernike_moments**2
    
    # determine the total number of descriptors based on the order N
    n = np.asarray(range(1,N+2))
    F_nl = ( n - ( (-1) + (-1)**n )/2 )/2
    zernike_descriptor = np.zeros(sum(F_nl))
    
    # sum the squared moments for each zernike descriptor 
    omega_nlm = 0
    for n in range(0,N+1):
        for l in range(n+1): # l goes from 0 to n
            for m in range(-l,l+1): # m goes from -l to l
                zernike_descriptor[F_nl] += sq_moments["indexing"]
            omega_nlm += 1
    
    
    
#%%

    
    
    
    
    
    
    
