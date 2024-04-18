# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:46:38 2023

@author: Emily Baum
"""

#%%

def read_ply(file_name,n_edges=3):
    
    """   
    
    Parameters
    ----------
    file_name : file.ply
    
        Protein Connolly surface file in .ply format (output file from EDTsurf)
        
    n_edges : int, optional
        
        Number of edges on each face of the mesh.
        The default is 3. (triangular mesh)

    Returns
    -------
    vertices : numpy array of float
    
        Points on the Connolly surface (x,y,z) and color code (r,g,b) from EDTsurf.
        Recommended color based on nearest atom.
    
    faces : numpy array of float
        
        Information used by ply files to create a triangular mesh.
        [n_edges=3,point_index_1:n_edges,r,g,b]
    
    """
    
    import numpy as np
    
    # read file
    with open(f"{file_name}") as f:
        text = f.read()
        
    # remove header, spaces, new line characters
    text = text.split('end_header') 
    text = text[1].strip()
    text = text.split('\n') 
    
    # save vertex and face information
    vertices = []
    faces = []
    for i in range(len(text)):
        f = text[i].split()
        f = [float(n) for n in f]
        if len(f) == n_edges+3:
            vertices.append(f) # rows with vertex information
        else:
            faces.append(f) # rows with face information

    return vertices, np.asarray(faces)


#%%

def make_surface_patches(surface_points,n_patches=1,patch_radius=8,features=True,atom_centers=None):
    
    """
    
    Parameters
    ----------
    surface_points : numpy array of float
        
        Coordinates, color code, (and opt. custom features) of protein surface points.
        Recommended to use vertices output from read_ply().
        
    n_patches : int, optional
        
        Number of unique protein surface patches to generate. The default is 1.
    
    patch_radius : float, optional
    
        The radius of the generated protein patch in Angstroms. The default is 8.
        
    features: bool, optional
    
        Include pre-defined features along with surface coordinates.
        The default is True.
    
    atom_centers : file.pdb, optional
    
        The protein file from which to include atom center information.
        This may be desired to distinguish a patch surface from its inverse.
        Using this with features = True may cause issues downstream unless features
        for atom center coordinates are added manually. The default is None.

    Returns
    -------
    patch_dict : dict of numpy arrays
    
        The dictionary contains the coordinates of all surface points (and opt
        atom centers) within patch_radius of a random surface point.
        Dictionary entries are named by the centerpoint_index of each patch.
        If features = True, each dictionary entry will contain an array of coordinates
        and an array of features.

    """
    
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42) # seed=42 for reproducibility
    # choose random patch centers (no duplicates)
    patch_center = rng.choice(len(surface_points), size=n_patches, replace=False)
    
    patch_dict = {}
    
    for centerpoint_index in patch_center:
        
        # initialize patch with centerpoint
        patch = [surface_points[centerpoint_index][:3]]
        if features == True:
            feat = [surface_points[centerpoint_index][3:]]
        
        # calculate which points are within the patch radius
        for i in surface_points: 
            if i == surface_points[centerpoint_index]:
                continue
            r = ((surface_points[centerpoint_index][0] - i[0])**2 + 
                 (surface_points[centerpoint_index][1] - i[1])**2 + 
                 (surface_points[centerpoint_index][2] - i[2])**2)**.5
            
            # add point coordinates to patch
            if r <= patch_radius:
                patch.append(i[:3])
                
                # (opt. - add features to patch)
                if features == True:
                    feat.append(i[3:])
        
        # calculate which atom centers are within the patch radius
        if atom_centers is not None:
            atoms = get_atom_centers(f"{atom_centers}")
            for i in atoms: 
                r = ((surface_points[centerpoint_index][0] - i[0])**2 + 
                     (surface_points[centerpoint_index][1] - i[1])**2 + 
                     (surface_points[centerpoint_index][2] - i[2])**2)**.5
                
                # add atom center coordinates to patch
                if r <= patch_radius:
                    patch.append(i)
        
        patch = np.asarray(patch)
        feat = np.asarray(feat)
        
        patch_dict[centerpoint_index] = patch, feat
        
    return patch_dict

#%%
def get_atom_centers(file_name):
    
    """

    Parameters
    ----------
    file_name : file.pdb
    
        The protein file from which to extract atom center information.

    Returns
    -------
    atom_centers : numpy array of float
        
        The coordinates (x,y,z) of each atom in the protein structure (including hydrogens).
        Water molecules and heteroatoms (such as ions and small molecules) are removed.

    """
    
    import numpy as np
    from Bio.PDB.PDBParser import PDBParser
    
    # obtain atom information from pdb file
    parser = PDBParser()  
    structure = parser.get_structure(f"{file_name}", f"{file_name}")  
    atoms = structure.get_atoms()
    
    # obtain all atom coordinates from structure
    atom_centers = []
    for atom in atoms:
        if 'W' not in atom.full_id[3][0]: # remove waters
            if 'H_' not in atom.full_id[3][0]: # remove heteroatoms (ions and small molecules)
                atom_centers.append(atom.get_coord())
    atom_centers = np.asarray(atom_centers)
    
    return atom_centers

#%%

def scale_patch(patch_dict,patch_radius=8,features=True):
    
    """

    Parameters
    ----------
    patch_dict : dict of numpy arrays
    
        The dictionary contains the coordinates of all surface points (and opt
        atom centers) within patch_radius of a random surface point.
        Dictionary entries are named by the centerpoint_index of each patch.
        
    patch_radius : float, optional
    
        The radius of the generated protein patch in Angstroms. The default is 8.
        
    features: bool, optional
    
        Indicate whether patch_dict was created with pre-defined features along
        with surface coordinates. The default is True.

    Returns
    -------
    scaled_dict : dict of numpy arrays
    
        The dictionary contains the scaled coordinates of all surface points
        (and opt atom centers) within patch_radius of a random surface point.
        Dictionary entries are named by the centerpoint_index of each patch.
        Coordinates must be scaled into a unit sphere for the 3D Zernike function.

    """
    
    import numpy as np
    
    scaled_dict = {}
    
    for patch in patch_dict.keys():
        
        # adjust indexing if features are included in dictionary
        if features == True:
            coordinates = patch_dict[patch][0]
        else:
            coordinates = patch_dict[patch]
        
        # scale patch coordinates by radius
        scaled_patch = []
        for point in coordinates:
            scaled_point = (point - coordinates[0])/patch_radius
            scaled_patch.append(scaled_point)
        scaled_patch = np.asarray(scaled_patch)
        
        # adjust indexing if features are included in dictionary
        if features == True:
            scaled_dict[patch] = scaled_patch, patch_dict[patch][1]
        else:
            scaled_dict[patch] = scaled_patch

    return scaled_dict
