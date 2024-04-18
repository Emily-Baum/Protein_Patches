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
    voxels : numpy array of float
    
        The coordinates of the points at which to calculate the value of the 
        3D Zernike function. A voxelized grid of evenly spaced points is recommended
        so that the polynomials are defined across the entire unit sphere.
        
    N : int, optional
    
        Order of approximation of the 3D Zernike function. Higher order provides
        higher resolution but increases computational time. Choosing an order too
        large may lead to overfitting. Must be consistent across polynomials,
        moments, and descriptors. The default is 20.

    Returns
    -------
    Z : numpy array of float
    
        Calculates the value of the 3D Zernike function at a given set of coordinates 
        within a unit sphere. Equations numbers follow: https://doi.org/10.1002/prot.22030.

    """
    
    import math
    import numpy as np
    
    # radius of point from patch center
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

def voxelize(scaled_patch,gridpoints=20,feature_array=None):
    
    """
    
    Parameters
    ----------
    scaled_patch : numpy array of float
    
        The scaled coordinates of all surface points (and opt atom centers) within 
        patch_radius of a random surface point.
        
    gridpoints : int, optional
    
        The number of voxels in each dimension to construct a unit grid (gridpoints**3).
        Only the voxels within a unit sphere are retained. The default is 20.

    feature_array: numpy array of float, optional
    
        An array of features to map to the voxelized grid. The default is None.
        
    Returns
    -------
    voxels : list of float
    
        The coordinates of each voxel certerpoint.
        
    voxel_surf: numpy array of int
    
        The mapping of the surface patch to the voxel grid. The values in the 
        list are 1 if the voxel contains part of the surface, or 0 if it does not.
        
    voxel_feat: numpy array of float, optional
    
        The values of the features defined in feature_array for each voxel. Empty voxels
        are given a default value of 0, and voxels containing multiple surface points
        are assigned an average value.

    """
    
    import numpy as np
    
    # create a unit grid with gridpoints**3 points, saving only those inside unit sphere
    x = np.linspace(-1,1,gridpoints)
    voxels = [[i,j,k] for i in x for j in x for k in x if i**2 + j**2 + k**2 <= 1]
    
    # map the surface patch to the voxelized coordinate grid
    voxel_surf = voxel_surface(scaled_patch,voxels,gridpoints)
    
    # if a feature array is specified, map the features to the voxels
    if feature_array is None:
        return voxels, voxel_surf
    else:
        voxel_feat = voxel_features(scaled_patch,feature_array,voxels,gridpoints)
        return voxels, voxel_surf, voxel_feat

#%%

def voxel_surface(scaled_patch,voxels,gridpoints):
    
    """
    
    Parameters
    ----------
    scaled_patch : numpy array of float
    
        The scaled coordinates of all surface points (and opt atom centers) within 
        patch_radius of a random surface point.
        
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
    
    # calculate the spacing between each voxel
    # half the distance between each voxel center
    voxel_spacing = 1/(gridpoints-1)
    
    # check if any patch points are within each cubic voxel
    voxel_surf = np.zeros(len(voxels))
    for v_coords in voxels:
        for p_coords in scaled_patch:
            if abs(v_coords[0] - p_coords[0]) < voxel_spacing:
                if abs(v_coords[1] - p_coords[1]) < voxel_spacing:
                    if abs(v_coords[2] - p_coords[2]) < voxel_spacing:
                        
                        # if a surface point falls within the voxel, mark the voxel as full
                        voxel_surf[voxels.index(v_coords)] = 1
                        continue
                    
    return voxel_surf

#%%
def voxel_features(scaled_patch,feature_array,voxels,gridpoints):
    
    """
    
    Parameters
    ----------
    scaled_patch : numpy array of float
    
        The scaled coordinates of all surface points (and opt atom centers) within 
        patch_radius of a random surface point.
        
    feature_array: numpy array of float
    
        An array of features to map to the voxelized grid.
        
    voxels : list of float
    
        The coordinates of each voxel certerpoint.
        
    gridpoints : int
    
        The number of voxels in each dimension to construct a unit grid (gridpoints**3).
        Only the voxels within a unit sphere are retained.

    Returns
    -------
    voxel_feat : numpy array of float
    
        The mapping of the surface patch features to the voxel grid. Empty voxels
        are assigned a default value of 0. Voxels containing multiple surface points
        are assigned the average feature value within the voxel.

    """

    import numpy as np
    
    # calculate the spacing between each voxel
    # half the distance between each voxel center
    voxel_spacing = 1/(gridpoints-1)
    
    # check if any patch points are within each cubic voxel
    voxel_feat = np.zeros((len(voxels),np.shape(feature_array)[1]))
    for v_coords in voxels:
        
        # count number of surface points in voxel
        n_in_vox = 0
        for p_coords in range(len(scaled_patch)):
            if abs(v_coords[0] - scaled_patch[p_coords][0]) < voxel_spacing:
                if abs(v_coords[1] - scaled_patch[p_coords][1]) < voxel_spacing:
                    if abs(v_coords[2] - scaled_patch[p_coords][2]) < voxel_spacing:
                        
                        # if a surface point falls within the voxel, add feature value
                        voxel_feat[voxels.index(v_coords)] += feature_array[p_coords]
                        n_in_vox += 1
        
        if n_in_vox == 0:
            # leave default empty voxel value as 0
            continue
        else:
            # take the average value in the voxel
            voxel_feat[voxels.index(v_coords)] /= n_in_vox
            
    return voxel_feat

#%% 
def zernike_moments(z_polynomials,voxel_surf,gridpoints=20,N=20):
    
    """
    
    Parameters
    ----------
    z_polynomials : numpy array of float
    
        The value of the 3D Zernike function at a given set of coordinates
        within a unit sphere.
        
    voxel_surf : numpy array of float
    
        The mapping of the surface patch to the voxel grid. The values in the 
        list are 1 if the voxel contains part of the surface, or 0 if it does not.
        Alternatively, an array of values for mapped features can be used as input.
        
    gridpoints : int, optional
    
        The number of voxels in each dimension to construct a unit grid (gridpoints**3).
        Only the voxels within a unit sphere are retained. The default is 20.
        
    N : int, optional
    
        Order of approximation of the 3D Zernike function. Higher order provides
        higher resolution but increases computational time. Choosing an order too
        large may lead to overfitting. Must be consistent across polynomials,
        moments, and descriptors. The default is 20.

    Returns
    -------
    corrected_moments : numpy array of float
    
        Calculates the 3D Zernike moments, necessary for computation of the Zernike 
        descriptors. Equation numbers follow: https://doi.org/10.1145/781606.781639

    """
        
    import numpy as np
    
    zernike_fit = np.linalg.lstsq(z_polynomials,voxel_surf,rcond=None)[0]
    
    voxel_spacing = 1/(gridpoints-1)
    
    # eqn. 8.2 (omega as a function of f)
    moments = 3/(4*np.pi)*zernike_fit*(voxel_spacing)**3
    
    # following the symmetry relation in eqn. 9
    corrected_moments = np.real(moments)
    omega_index = 0
    
    # index which moments come from -m terms
    for n in range(0,N+1):
        for l in range(n+1): # l goes from 0 to n
            if ((n-l)%2) == 0: # (n-l) even
                for m in range(-l,l+1): # m goes from -l to l
                    if m < 0:
                        pos_m = int(omega_index + -2*m)
                        
                        # eqn. 9
                        corrected_moments[omega_index] = corrected_moments[pos_m]*(-1)**m
                    omega_index += 1
                    
    return corrected_moments
    
#%%
def zernike_descriptor(moments,N=20):
    
    """

    Parameters
    ----------
    moments : numpy array of float
    
        The 3D Zernike moments for a patch voxelized within a unit sphere.
        The moments can contain geometric or feature information.
        
    N : int, optional
    
        Order of approximation of the 3D Zernike function. Higher order provides
        higher resolution but increases computational time. Choosing an order too
        large may lead to overfitting. Must be consistent across polynomials,
        moments, and descriptors. The default is 20.

    Returns
    -------
    zernike_descriptor : numpy array of float
    
        A compact, non-redundant, rotationally invariant descriptor used to 
        represent protein surface patches.
        Equations numbers follow: https://doi.org/10.1002/prot.22030.

    """
    
    import numpy as np
    sq_moments = moments**2
    
    # determine the total number of descriptors based on the order N
    n = np.asarray(range(1,N+2))
    F_nl = ( n - ( (-1) + (-1)**n )/2 )/2
    
    # creating an array of length F_nl, following eqn. 10
    moment_sum = np.zeros(int(sum(F_nl)))
    
    # sum the squared moments for each zernike descriptor 
    omega_index = 0
    zd_index = 0
    for n in range(0,N+1):
        for l in range(n+1): # l goes from 0 to n
            if ((n-l)%2) == 0: # (n-l) even
                
                # sum over m for each combination of n and l
                for m in range(-l,l+1): # m goes from -l to l
                    moment_sum[zd_index] += sq_moments[omega_index]
                    omega_index += 1
                zd_index += 1
    
    # eqn. 10
    zernike_descriptor = moment_sum**.5
    return zernike_descriptor
