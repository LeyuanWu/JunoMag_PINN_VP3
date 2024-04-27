# %% # ! Import
import numpy as np;

def MD2ecef(MDx,MDy,MDz,md_lon,md_lat,systemIII=True):
    """Magnetic Dipole to ECEF Cartesian coordinates
    Args:
        MDx,MDy,MDz (numpy array):  Magnetic Dipole coordinates
        md_lon: Magnetic dipole Longitude
        md_lat: Magnetic dipole Latitude
        systemIII (bool, optional): . Defaults to True.

    Returns:
        x,y,z: (numpy array):  ECEF coordinates
    """
    if(systemIII):
        md_lon = 360.0-md_lon;
    md_lon = np.deg2rad(md_lon);
    md_lat = np.deg2rad(md_lat);
    unitMDx = np.array([np.cos(md_lat)*np.cos(md_lon),np.cos(md_lat)*np.sin(md_lon),-np.sin(md_lat)]).reshape(1,-1);
    unitMDy = np.array([-np.sin(md_lon),np.cos(md_lon),0]).reshape(1,-1);
    unitMDz = np.array([np.sin(md_lat)*np.cos(md_lon),np.sin(md_lat)*np.sin(md_lon),np.cos(md_lat)]).reshape(1,-1);
    MD_Pts = np.hstack((MDx,MDy,MDz));
    Pts = MD_Pts@np.vstack((unitMDx, unitMDy, unitMDz));
    return Pts[:,0:1], Pts[:,1:2], Pts[:,2:3];

def MD2ecef_v(MDu,MDv,MDw,md_lon,md_lat,systemIII=True):
    """Magnetic Dipole to ECEF Cartesian vectors
    Args:
        MDu,MDv,MDw (numpy array):  Magnetic Dipole vector components
        md_lon: Magnetic dipole Longitude
        md_lat: Magnetic dipole Latitude
        systemIII (bool, optional): . Defaults to True.
    Returns:
        u,v,w,norm: (numpy array):  ECEF vector components and magnitude
    """
    if(systemIII):
        md_lon = 360.0-md_lon;
    md_lon = np.deg2rad(md_lon);
    md_lat = np.deg2rad(md_lat);
    unitMDx = np.array([np.cos(md_lat)*np.cos(md_lon),np.cos(md_lat)*np.sin(md_lon),-np.sin(md_lat)]).reshape(1,-1);
    unitMDy = np.array([-np.sin(md_lon),np.cos(md_lon),0]).reshape(1,-1);
    unitMDz = np.array([np.sin(md_lat)*np.cos(md_lon),np.sin(md_lat)*np.sin(md_lon),np.cos(md_lat)]).reshape(1,-1);
    MD_VECs = np.hstack((MDu,MDv,MDw));
    VECs = MD_VECs@np.vstack((unitMDx, unitMDy, unitMDz));
    Norms = np.sqrt(np.sum(VECs**2,axis=1,keepdims=True));
    return VECs[:,0:1], VECs[:,1:2], VECs[:,2:3], Norms;


def ecef2MD(x,y,z,md_lon,md_lat,systemIII=True):
    """ECEF to Magnetic Dipole Cartesian coordinates
    Args:
        x,y,z (numpy array):  ECEF coordinates
        md_lon: Magnetic dipole Longitude
        md_lat: Magnetic dipole Latitude
        systemIII (bool, optional): . Defaults to True.
    Returns:
        MDx,MDy,MDz: (numpy array):  Magnetic Dipole coordinates
    """
    if(systemIII):
        md_lon = 360.0-md_lon;
    md_lon = np.deg2rad(md_lon);
    md_lat = np.deg2rad(md_lat);
    unitMDx = np.array([np.cos(md_lat)*np.cos(md_lon),np.cos(md_lat)*np.sin(md_lon),-np.sin(md_lat)]).reshape(-1,1);
    unitMDy = np.array([-np.sin(md_lon),np.cos(md_lon),0]).reshape(-1,1);
    unitMDz = np.array([np.sin(md_lat)*np.cos(md_lon),np.sin(md_lat)*np.sin(md_lon),np.cos(md_lat)]).reshape(-1,1);
    Pts = np.hstack((x,y,z));
    MD_Pts = Pts@np.hstack((unitMDx, unitMDy, unitMDz));
    return MD_Pts[:,0:1], MD_Pts[:,1:2], MD_Pts[:,2:3];

def ecef2MD_v(u,v,w,md_lon,md_lat,systemIII=True):
    """ECEF to Magnetic Dipole Cartesian vectors
    Args:
        (u,v,w) (numpy array):  ECEF vector components
        md_lon: Magnetic dipole Longitude
        md_lat: Magnetic dipole Latitude
        systemIII (bool, optional): . Defaults to True.
    Returns:
        MDu,MDv,MDw,MD_Norm: (numpy array):  Magnetic Dipole vector components and magnitude
    """
    if(systemIII):
        md_lon = 360.0-md_lon;
    md_lon = np.deg2rad(md_lon);
    md_lat = np.deg2rad(md_lat);
    unitMDx = np.array([np.cos(md_lat)*np.cos(md_lon),np.cos(md_lat)*np.sin(md_lon),-np.sin(md_lat)]).reshape(-1,1);
    unitMDy = np.array([-np.sin(md_lon),np.cos(md_lon),0]).reshape(-1,1);
    unitMDz = np.array([np.sin(md_lat)*np.cos(md_lon),np.sin(md_lat)*np.sin(md_lon),np.cos(md_lat)]).reshape(-1,1);
    VECs = np.hstack((u,v,w));
    MD_VECs = VECs@np.hstack((unitMDx, unitMDy, unitMDz));
    MD_Norm = np.sqrt(np.sum(MD_VECs**2,axis=1,keepdims=True));
    return MD_VECs[:,0:1], MD_VECs[:,1:2], MD_VECs[:,2:3], MD_Norm;


def ell2ecef(Lon,Lat,H,a,ecc):
    """Ellipsoidal to ECEF Cartesian coordinates
    Args:
        Lon: longitudes in radians
        Lat: latitudes in radians
        H: ellipsoidal height in meters
        a: major axis of sphroid in meters
        ecc: eccentricity

    Returns:
        X,Y,Z: ECEF Cartesian coordinates
    """
    e2 = ecc**2;
    N = a/np.sqrt(1-e2*np.sin(Lat)**2);
    X = (N+H)*(np.cos(Lat)*np.cos(Lon));
    Y = (N+H)*(np.cos(Lat)*np.sin(Lon));
    Z = (N*(1-e2)+H)*np.sin(Lat);
    return X, Y, Z;

def ecef2sph(X,Y,Z,colat=True,degrees=False):
    """From Earth-Centered-Earth-Fixed to 
        Spherical coordinates 
    Args:
        X,Y,Z (numpy array): ECEF Coordinates
        colat (bool, optional): Co-Latitude. Defaults to True.
        degrees (bool, optional): Degree or Radian. Defaults to False.

    Returns:
        R,Lat,Lon: Spherical Coordinates
    """
    R=np.sqrt(X**2+Y**2+Z**2);
    Lat=np.arccos(Z/R); # ! Co-latitude in [0,pi]
    Lon=np.arctan2(Y,X);
    # Lat to co-Lat
    if not colat:
        Lat=np.pi/2-Lat;
    # From radians to degrees
    if degrees:
        Lon = np.rad2deg(Lon);
        Lat = np.rad2deg(Lat);
    return R,Lat,Lon;

def ecef2ell(X,Y,Z,a,ecc,colat=True,degrees=False):
    """From Earth-Centered-Earth-Fixed to 
        Ellipsoidal (geodetic) coordinates using Bowring Algorithm
    Reference:
    Bowring, B.R., 1976. Transformation from spatial to geographical coordinates.
    Surv. Rev. 23 (181), 323-327.

    Args:
        X,Y,Z (numpy array): ECEF Coordinates
        a (float): semi-major
        ecc (float): eccentricity
        colat (bool, optional): Co-Latitude. Defaults to True.
        degrees (bool, optional): Degree or Radian. Defaults to False.

    Returns:
        Lon,Lat,H: Geodetic Coordinates
    """
    ecc2 = ecc**2;         # eccentricity squared
    b = a*np.sqrt(1-ecc2);   # semi-minor axis   
    Lon = np.arctan2(Y,X);   # Geographic longitude
    # Auxiliary values
    p = np.sqrt(X**2 + Y**2);
    mu = np.arctan(a*Z/(b*p));
    ea2 = ecc2/(1-ecc2);
    # Geographic latitude
    Lat = np.arctan2(Z + ea2*b*np.sin(mu)**3, p - ecc2*a*np.cos(mu)**3);
    # Height above ellipsoid
    H = p/np.cos(Lat) - a/np.sqrt(1-ecc2*np.sin(Lat)**2);
    # Lat to co-Lat
    if colat:
        Lat=np.pi/2-Lat;
    # from radians to degrees
    if degrees:
        Lon = np.rad2deg(Lon);
        Lat = np.rad2deg(Lat);
    return Lon,Lat,H;

def ecef2ned_v(X,Y,Z,U,V,W,a,ecc):
    """Transform vector from Earth-Centered-Earth-Fixed XYZ-components
    to Ellipsoidal (geodetic) North-East-Downward (NED) components 

    Args:
        X,Y,Z (numpy array): ECEF Coordinates
        U,V,W (numpy array): ECEF Vector components
        a (float): semi-major
        ecc (float): eccentricity

    Returns:
        Gn,Ge,Gd: North-East-Downward (NED) components
    """
    assert(X.shape==Y.shape==Z.shape==U.shape==V.shape==W.shape);
    shpComp=X.shape;
    X=X.reshape(-1,1);Y=Y.reshape(-1,1);Z=Z.reshape(-1,1);
    U=U.reshape(-1,1);V=V.reshape(-1,1);W=W.reshape(-1,1);
    if ecc==0:
        _,Lat,Lon = ecef2sph(X,Y,Z,colat=False);
    else:
        Lon,Lat,_ = ecef2ell(X,Y,Z,a,ecc,colat=False);

    unitx=np.array([1,0,0]).reshape(-1,1);
    unity=np.array([0,1,0]).reshape(-1,1);
    unitz=np.array([0,0,1]).reshape(-1,1);

    N=np.hstack((-np.sin(Lat)*np.cos(Lon),-np.sin(Lat)*np.sin(Lon),np.cos(Lat)));
    E=np.hstack((-np.sin(Lon),np.cos(Lon),np.zeros((Lon.shape))));
    D=np.hstack((-np.cos(Lat)*np.cos(Lon),-np.cos(Lat)*np.sin(Lon),-np.sin(Lat)));

    Gn=U*(N@unitx)+V*(N@unity)+W*(N@unitz);
    Ge=U*(E@unitx)+V*(E@unity)+W*(E@unitz);
    Gd=U*(D@unitx)+V*(D@unity)+W*(D@unitz);
    
    Gn=np.reshape(Gn,shpComp);
    Ge=np.reshape(Ge,shpComp);
    Gd=np.reshape(Gd,shpComp);
    
    return Gn,Ge,Gd;

def ned2ecef_v(Lat,Lon,Gn,Ge,Gd,colat=True,degrees=False):
    """Transform vector from 
    Ellipsoidal (geodetic) North-East-Downward (NED) components 
    to Earth-Centered-Earth-Fixed XYZ-components

    Args:
        Lat,Lon (numpy array): Lat and Lon coordinates
        Gn,Ge,Gd (numpy array): (NED) components
        colat (bool, optional): Co-Latitude. Defaults to True.
        degrees (bool, optional): Degree or Radian. Defaults to False.
    Returns:
        Gx,Gy,Gz: ECEF Vector components
    """
    assert(Lat.shape==Lon.shape==Gn.shape==Ge.shape==Gd.shape);
    shpComp=Lat.shape;
    Lat=Lat.reshape(-1,1);Lon=Lon.reshape(-1,1);
    Gn=Gn.reshape(-1,1);Ge=Ge.reshape(-1,1);Gd=Gd.reshape(-1,1);
    if degrees:
        Lon = np.deg2rad(Lon);
        Lat = np.deg2rad(Lat);
    if colat:
        Lat=np.pi/2-Lat;

    unitx=np.array([1,0,0]).reshape(-1,1);
    unity=np.array([0,1,0]).reshape(-1,1);
    unitz=np.array([0,0,1]).reshape(-1,1);

    N=np.hstack((-np.sin(Lat)*np.cos(Lon),-np.sin(Lat)*np.sin(Lon),np.cos(Lat)));
    E=np.hstack((-np.sin(Lon),np.cos(Lon),np.zeros((Lon.shape))));
    D=np.hstack((-np.cos(Lat)*np.cos(Lon),-np.cos(Lat)*np.sin(Lon),-np.sin(Lat)));

    Gx=Gn*(N@unitx)+Ge*(E@unitx)+Gd*(D@unitx);
    Gy=Gn*(N@unity)+Ge*(E@unity)+Gd*(D@unity);
    Gz=Gn*(N@unitz)+Ge*(E@unitz)+Gd*(D@unitz);
    
    Gx=np.reshape(Gx,shpComp);
    Gy=np.reshape(Gy,shpComp);
    Gz=np.reshape(Gz,shpComp);
    
    return Gx,Gy,Gz;