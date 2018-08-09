import numpy as np

## Handeling Coordinate System Changes

def dot3(p1,p2):
    return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]

def norm(p):
    return np.sqrt(dot3(p,p))

def hat(p):
    return [p[i]/norm(p) for i in [0,1,2]]

def cos_theta(p1,p2):
    return dot3(p1,p2) / norm(p1) / norm(p2)

def angle(p1,p2):
    cos = cos_theta(p1,p2)
    if np.isclose(cos, 1., 1e-15, 1e-15):
        theta = 0.
    elif np.isclose(cos, -1., 1e-15, 1e-15):
        theta = np.pi
    else: 
        theta = np.arccos(cos)
    return theta

def cross(p1,p2):
    c0 = p1[1]*p2[2] - p1[2]*p2[1]
    c1 = p1[2]*p2[0] - p1[0]*p2[2]
    c2 = p1[0]*p2[1] - p1[1]*p2[0]
    return [c0,c1,c2]

def compute_phi(p, ref):
    
    if np.isclose(angle(p, ref), 0., 1e-15, 1e-15):
        return 0.0
    
    # Define x,y,z directions by a convention
    e_z = np.asarray(hat(ref))
    e_x = np.asarray(hat(cross([0,1,0], e_z)))
    e_y = np.asarray(cross(e_z, e_x))

    # Project p into xy plane
    p3 = np.asarray(p[:3])    
    p2 = p3 - e_z * dot3(e_z, p3)
    
    #Determine azimuthal angle in xy plane
    phi_y = angle(p2, e_y)
    phi = angle(p2, e_x)
    if phi_y > np.pi/2 :
        phi = 2*np.pi - phi
    if np.isnan(phi) :
        phi = 0.0
    return phi

def EPT(pXYZT, ref):
    e = pXYZT[3]
    theta = angle(pXYZT, ref)
    phi = compute_phi(pXYZT, ref)
    return [e, theta, phi]

def XYZ(pEPT):
    e = pEPT[0]
    theta = pEPT[1]
    phi = pEPT[2]
    p_x = e * np.sin(theta) * np.cos(phi)
    p_y = e * np.sin(theta) * np.sin(phi)
    p_z = e * np.cos(theta)
    return [p_x, p_y, p_z]
    
def XYZ_not_normalized(pEPT):
    theta = pEPT[1]
    phi = pEPT[2]
    p_x = np.sin(theta) * np.cos(phi)
    p_y = np.sin(theta) * np.sin(phi)
    p_z = np.cos(theta)
    return [p_x, p_y, p_z]

def PT(pXYZT, ref):
    theta = angle(pXYZT, ref)
    phi = compute_phi(pXYZT, ref)
    return [theta, phi]

def computeUnitVectors(ref):
    e_z = np.asarray(hat(ref))
    e_x = np.asarray(hat(cross([0,1,0], e_z)))
    e_y = np.asarray(cross(e_z, e_x))
    return [e_x,e_y,e_z]

def relative_to_absolute_frame_ETP(daugther,parent):
    # Daugther momentum relative to parent's XYZ coordinate system
    daugtherRel = hat(XYZ_not_normalized(daugther))
    # Parent's coordinate system in terms of jet fram coordinates
    unitVectors = computeUnitVectors(hat(XYZ_not_normalized(parent)))
    # Convert to jet frame XYZ
    jetFrameXYZ = sum([daugtherRel[i]*unitVectors[i] for i in range(3)])
    PTvec = PT(hat(jetFrameXYZ),[0,0,1])
    return [daugther[0],PTvec[0],PTvec[1]]

def relative_to_absolute_frame_XYZ(daugther,parent):
    # Daugther momentum relative to parent's XYZ coordinate system
    daugtherRel = XYZ(daugther)
    # Parent's coordinate system in terms of jet fram coordinates
    unitVectors = computeUnitVectors(XYZ(parent))
    # Convert to jet frame XYZ
    jetFrameXYZ = sum([daugtherRel[i]*unitVectors[i] for i in range(3)])
    return jetFrameXYZ