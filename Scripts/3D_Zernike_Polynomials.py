# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 03:08:46 2023

@author: Emily
"""

#%%

def zernike3d(x,y,z,N=20):
    
    """

    Parameters
    ----------
    x : float
    
        The x coordinate of the point at which to calculate the value of the 
        3D Zernike function.
        
    y : float
        
        The y coordinate of the point at which to calculate the value of the 
        3D Zernike function.
        
    z : float
    
        The z coordinate of the point at which to calculate the value of the 
        3D Zernike function.
        
    N : TYPE
    
        Order of approximation of the 3D Zernike function. Higher order provides
        higher resolution but scales computation exponentially. The default is 20.

    Returns
    -------
    Z : list of float
    
        Calculates the value of the 3D Zernike function at a given coordinate within a unit sphere.
        Equations numbers follow: https://doi.org/10.1002/prot.22030.

    """
    
    import math

    r = (x**2 + y**2 + z**2)**.5
    Z = []
    
    for n in range(N+1): # n goes from 0 to order N
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
                        
                        usum += math.comb(l,u)*comb*((-(x**2 + y**2)/(4*z**2))**u)
                    
                    # eqn 5
                    elm = rl*clm*(((1j*x-y)/2)**m)*(z**(l-m))*usum 
                    # eqn 7
                    Znlm = elm*qsum 
                    Z.append(Znlm)
    return Z
        
