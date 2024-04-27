# %% # ! Import
import pyshtools as pysh;
import numpy as np;
from LW_CoordinateTransformation import *;
import time;

def Format_SHCsFile(oneCol_SHCsFile,nmax,mySHCsFile,internal=True):
    """Read 1-column SHC file and re-format
    Args:
        oneCol_SHCsFile (str): file contain SHCs
        nmax (int): Maximum degree of expansion
        mySHCsFile (str): output file
        internal: internal | external SH coefficients
    """
    ######## Load
    if(isinstance(oneCol_SHCsFile,str)):
        SHCs = np.loadtxt(oneCol_SHCsFile,skiprows=0);
    else:
        SHCs = oneCol_SHCsFile;
    ######## Save
    fid = open(mySHCsFile, 'w');
    if internal:
        fid.write('%9s %15s %9s %9s %9s'%('IND','SHC','g | h','n','m'));
    else:
        fid.write('%9s %15s %9s %9s %9s'%('IND','SHC','G | H','n','m'));
    ind=0;
    for n in range(1,nmax+1):
        for m in range(0,n+1):
            if internal:
                fid.write('\n%9d %15.6e %9s %9d %9d'%(ind+1,SHCs[ind],'g',n,m));
            else:
                fid.write('\n%9d %15.6e %9s %9d %9d'%(ind+1,SHCs[ind],'G',n,m));
            ind=ind+1;
            if m>0:
                if internal:
                    fid.write('\n%9d %15.6e %9s %9d %9d'%(ind+1,SHCs[ind],'h',n,m));
                else:
                    fid.write('\n%9d %15.6e %9s %9d %9d'%(ind+1,SHCs[ind],'H',n,m));
                ind=ind+1;
    fid.close();
    print('Saving SHCs to file: %s'%(mySHCsFile),flush=True);
    print('\n*********************\n',flush=True);
    return None;

def SHS_Bxyz(fileSH,nmax,a,x,y,z,nmin=1,showinfo=True,internal=True):
    """ SH expansion to <Bx, By, Bz> and |B| 
    Args:
        fileSH (str): file contain SH Coefficients. 
            Store in <Chaos> order: h right after g for each degree and order
        nmax (int): Maximum degree of expansion
        a (float): Reference radius
        x,y,z (numpy array): 
            Cartesian coordinate of <Computation> points
    Units:
        B in Gauss
    Returns:
        Bn,Be,Bd: North-East-Downward Components 
    """
    RPs,ThetaPs,LambdaPs = ecef2sph(x,y,z,colat=True,degrees=False);
    Bn,Be,Bd = SHS_Bned(fileSH,nmax,a,RPs,ThetaPs,LambdaPs,nmin=nmin,showinfo=showinfo,internal=internal);
    Bx,By,Bz = ned2ecef_v(ThetaPs,LambdaPs,Bn,Be,Bd);
    BNorm=np.sqrt(Bx**2+By**2+Bz**2); 
    return Bx,By,Bz,BNorm;

def SHS_Bned(fileSH,nmax,a,RPs,ThetaPs,LambdaPs,nmin=1,showinfo=True,internal=True):
    """ SH expansion to <Bn, Be, Bd> 
    Args:
        fileSH: file contain SH Coefficients.  
            Store in <Chaos> order: h right after g for each degree and order
        nmax (int): Maximum degree of expansion
        a (float): Reference radius
        RPs,ThetaPs,LambdaPs (numpy array): 
            Radius, Co-latitude, longitude of <Computation> points
    Units:
        B in Gauss
    Returns:
        Bn,Be,Bd: North-East-Downward Components 
    """
    ######## ! Constants
    nT2Gauss=1e-5; # nT to Gauss
    ######## !  SH Expansion from a
    assert(RPs.shape==LambdaPs.shape==ThetaPs.shape);
    ######## ! Load SHCs from txt file
    ghSHCs = np.loadtxt(fileSH,skiprows=1,usecols=1);
    #### Preparation
    shpCmp=RPs.shape;
    RPs=RPs.reshape(-1,1);
    ThetaPs=ThetaPs.reshape(-1,1);
    LambdaPs=LambdaPs.reshape(-1,1);
    #### Store ALF and Derivative
    nCoef=int((nmax+1)*(nmax+2)/2);
    nObs=RPs.size;
    PnmThetaPs=np.zeros((nCoef,nObs));
    PnmThetaPs_d=np.zeros((nCoef,nObs));
    for iObs in range(nObs):
        PnmThetaPs[:,iObs], PnmThetaPs_d[:,iObs] \
            = pysh.legendre.PlmSchmidt_d1(nmax,np.cos(ThetaPs[iObs]));
    if showinfo:
        print('Expanding SH coefficients...%s'%(time.ctime()),flush=True);
    Btheta=np.zeros(RPs.shape); Blambda=np.zeros(RPs.shape); Br=np.zeros(RPs.shape);
    ind=nmin**2-1;
    for n in np.arange(nmin,nmax+1):
        if showinfo:
            print('n=%d \t'%(n),flush=True);
        if internal:
            nFacBn=(a/RPs)**(n+2);
            nFacBd=-(n+1)*(a/RPs)**(n+2);
        else:
            nFacBn=(RPs/a)**(n-1);
            nFacBd= n*(RPs/a)**(n-1);
        if showinfo:
            print('m=',end='',flush=True);
        for m in np.arange(0,n+1):
            if showinfo:
                print('%d,'%(m),end='',flush=True);
            indRow=int(n*(n+1)/2+m);
            tempPnm=PnmThetaPs[indRow,:].reshape(-1,1);
            tempPnm_d=-np.sin(ThetaPs)*PnmThetaPs_d[indRow,:].reshape(-1,1);
            gnm=ghSHCs[ind]; ind=ind+1;
            Btheta=Btheta+nFacBn*tempPnm_d*gnm*np.cos(m*LambdaPs);
            Blambda=Blambda+nFacBn/np.sin(ThetaPs)*tempPnm*(-m*gnm*np.sin(m*LambdaPs));
            Br=Br+nFacBd*tempPnm*(gnm*np.cos(m*LambdaPs));
            if m>0:
                hnm=ghSHCs[ind]; ind=ind+1;
                Btheta=Btheta+nFacBn*tempPnm_d*hnm*np.sin(m*LambdaPs);
                Blambda=Blambda+nFacBn/np.sin(ThetaPs)*tempPnm*(m*hnm*np.cos(m*LambdaPs));
                Br=Br+nFacBd*tempPnm*(hnm*np.sin(m*LambdaPs));
        if showinfo:
            print(''); # <br>
    if showinfo:
        print('Complete SH exanding at %s'%(time.ctime()),flush=True);
    Bn=(-1)*nT2Gauss*np.reshape(-Btheta,shpCmp); # B = - \nabla u
    Be=(-1)*nT2Gauss*np.reshape(Blambda,shpCmp);
    Bd=(-1)*nT2Gauss*np.reshape(-Br,shpCmp);
    return Bn, Be, Bd;

def design_Gauss(nmax,a,RPs,ThetaPs,LambdaPs,showinfo=True):
    """ Computes matrices to connect the radial, colatitude and azimuthal field components 
    to the magnetic potential field in terms of spherical harmonic coefficients (Schmidt quasi-normalized).
    Args:
        nmax (int): Maximum degree of expansion
        a (float): Reference radius
        RPs,ThetaPs,LambdaPs (numpy array): 
            Radius, Co-latitude, longitude of <Computation> points
    Returns:
        Gn,Ge,Gd: Gaussian Matrices using either North-East-Downward components
    """
    ######## ! Constants
    nT2Gauss=1e-5; # nT to Gauss
    ######## !  SH Expansion from a
    assert(RPs.shape==LambdaPs.shape==ThetaPs.shape);
    #### Preparation
    shpCmp=RPs.shape;
    RPs=RPs.reshape(-1,);
    ThetaPs=ThetaPs.reshape(-1,);
    LambdaPs=LambdaPs.reshape(-1,);
    #### Store ALF and Derivative
    nCoef=int((nmax+1)*(nmax+2)/2);
    nObs=RPs.size;
    PnmThetaPs=np.zeros((nCoef,nObs));
    PnmThetaPs_d=np.zeros((nCoef,nObs));
    for iObs in range(nObs):
        PnmThetaPs[:,iObs], PnmThetaPs_d[:,iObs] \
            = pysh.legendre.PlmSchmidt_d1(nmax,np.cos(ThetaPs[iObs]));
    ######## !  The Gauss matrices Gn, Ge, Gd
    if showinfo:
        print('Computing transform Gauss matrix ...%s'%(time.ctime()),flush=True);
    nCoeff_g=int((nmax+1)*(nmax+2)/2)-1;
    nCoeff_h=int(nmax*(nmax+1)/2);
    Gn=np.zeros((nObs,nCoeff_g+nCoeff_h));
    Ge=np.zeros((nObs,nCoeff_g+nCoeff_h));
    Gd=np.zeros((nObs,nCoeff_g+nCoeff_h));
    ind=0;
    for n in np.arange(1,nmax+1):
        if showinfo:
            print('n=%d \t'%(n),flush=True);
        nFacBn=(a/RPs)**(n+2);
        nFacBd=-(n+1)*(a/RPs)**(n+2);
        if showinfo:
            print('m=',end='',flush=True);
        for m in np.arange(0,n+1):
            if showinfo:
                print('%d,'%(m),end='',flush=True);
            indRow=int(n*(n+1)/2+m);
            tempPnm=PnmThetaPs[indRow,:];
            tempPnm_d=-np.sin(ThetaPs)*PnmThetaPs_d[indRow,:];
            Gn[:,ind] = nFacBn*tempPnm_d*np.cos(m*LambdaPs);
            Ge[:,ind] = (-1)*nFacBn/np.sin(ThetaPs)*tempPnm*(-m*np.sin(m*LambdaPs));
            Gd[:,ind] = nFacBd*tempPnm*(np.cos(m*LambdaPs));
            ind=ind+1;
            if m>0:
                Gn[:,ind] = nFacBn*tempPnm_d*np.sin(m*LambdaPs);
                Ge[:,ind] = (-1)*nFacBn/np.sin(ThetaPs)*tempPnm*(m*np.cos(m*LambdaPs));
                Gd[:,ind] = nFacBd*tempPnm*(np.sin(m*LambdaPs));
                ind=ind+1;
        if showinfo:
            print(''); # <br>
    if showinfo:
        print('Complete Gauss matrices Gn, Ge, Gd computation at %s'%(time.ctime()),flush=True);
    return nT2Gauss*Gn, nT2Gauss*Ge, nT2Gauss*Gd;

def LowesSpectrum(ghSHCs,nmax,radius):
    """Lowes Spectrum Computation
    Args:
        ghSHCs (numpy array): Store in <Chaos> order: h right after g for each degree and order
        g10,g11,h11,g20,g21,h21,g22,h22,... (3+5+7+9+...=N(N+2))
        nmax (int): Maximum degree of SH model
        radius: Computation radius other than 1 (surface)
    Returns:
        Rn (numpy array): Lowes Spectrum
    """
    Rn = np.zeros(nmax,)
    ind_coef = 0;
    for n in range(1,nmax+1):
        noc = 2*n+1;
        Rn[n-1] = (n+1)*(1/radius)**(2*n+4)*np.sum(ghSHCs[ind_coef:ind_coef+noc]**2)
        ind_coef = ind_coef+noc;
    return Rn;

def MagnetoDisc(Xs, Ys, Zs, R0, R1, D, muI_2):
    """Compute the magnetic field cause by a magnetodisc
    Args:
        Xs,Ys,Zs (numpy array N*1): Magnetic Dipote coordinates (Jovian radii)
        R0: Disc inner radius (Jovian radii)
        R1: Disc outer radius (Jovian radii)
        D: Half thickness (Jovian radii)
        muI_2: Current constant (nT)
    Returns:
        Bx, By, Bz, BNorm (numpy array N*1): Magnetic field components and magnitude
    """
    RHOs = np.sqrt(Xs**2+Ys**2);
    indA = (RHOs<=R0);
    indB = np.logical_and((RHOs>R0),(RHOs<R1));
    indC = (RHOs>R1);
    Bx = np.full(Xs.shape,np.nan); By = np.full(Xs.shape,np.nan); Bz = np.full(Xs.shape,np.nan);
    ######## Region A
    tBrhoA_1,tBzA_1 = SmallApprox(RHOs[indA],Zs[indA],R0,D,muI_2);
    tBxA_1 = tBrhoA_1*Xs[indA]/RHOs[indA];
    tByA_1 = tBrhoA_1*Ys[indA]/RHOs[indA];
    tBrhoA_2,tBzA_2 = SmallApprox(RHOs[indA],Zs[indA],R1,D,muI_2);
    tBxA_2 = tBrhoA_2*Xs[indA]/RHOs[indA];
    tByA_2 = tBrhoA_2*Ys[indA]/RHOs[indA];
    Bx[indA] = tBxA_1 - tBxA_2;
    By[indA] = tByA_1 - tByA_2;
    Bz[indA] = tBzA_1 - tBzA_2;
    ######## Region B
    tBrhoB_1,tBzB_1 = LargeApprox(RHOs[indB],Zs[indB],R0,D,muI_2);
    tBxB_1 = tBrhoB_1*Xs[indB]/RHOs[indB];
    tByB_1 = tBrhoB_1*Ys[indB]/RHOs[indB];
    tBrhoB_2,tBzB_2 = SmallApprox(RHOs[indB],Zs[indB],R1,D,muI_2);
    tBxB_2 = tBrhoB_2*Xs[indB]/RHOs[indB];
    tByB_2 = tBrhoB_2*Ys[indB]/RHOs[indB];
    Bx[indB] = tBxB_1 - tBxB_2;
    By[indB] = tByB_1 - tByB_2;
    Bz[indB] = tBzB_1 - tBzB_2;
    ######## Region A
    tBrhoC_1,tBzC_1 = LargeApprox(RHOs[indC],Zs[indC],R0,D,muI_2);
    tBxC_1 = tBrhoC_1*Xs[indC]/RHOs[indC];
    tByC_1 = tBrhoC_1*Ys[indC]/RHOs[indC];
    tBrhoC_2,tBzC_2 = LargeApprox(RHOs[indC],Zs[indC],R1,D,muI_2);
    tBxC_2 = tBrhoC_2*Xs[indC]/RHOs[indC];
    tByC_2 = tBrhoC_2*Ys[indC]/RHOs[indC];
    Bx[indC] = tBxC_1 - tBxC_2;
    By[indC] = tByC_1 - tByC_2;
    Bz[indC] = tBzC_1 - tBzC_2;

    BNorm = np.sqrt(Bx**2+By**2+Bz**2);
    return Bx, By, Bz, BNorm;

def SmallApprox(RHOs,Zs,R0,D,muI_2):
    """Compute the magnetic field cause by a magnetodisc (small rho approximation)
    Ref: Edwards, T. M., Bunce, E. J., & Cowley, S. W. H. (2001). 
         A note on the vector potential of Connerney et al.'s model of the 
         equatorial current sheet in Jupiter's magnetosphere. 
         Planetary and Space Science, 49, 1115-1123. https://doi.org/10.1016/S0032-0633(00)00164-1

    Args:
        RHOs,Zs: (numpy array (N,)): Magnetic Dipote coordinates (Jovian radii)
        R0: Disc inner radius (Jovian radii)
        D: Half thickness (Jovian radii)
        muI_2: Current constant (nT)

    Returns:
        Brho, Bz (numpy array (N,)): Magnetic field components and magnitude
    """
    p1 = RHOs/2*(1/(np.sqrt((Zs-D)**2+R0**2))-1/(np.sqrt((Zs+D)**2+R0**2)));
    p2 = RHOs**3/16*((R0**2-2*(Zs-D)**2)/(R0**2+(Zs-D)**2)**2.5 - (R0**2-2*(Zs+D)**2)/(R0**2+(Zs+D)**2)**2.5);
    Brho = muI_2*(p1+p2);
    q1 = np.log((Zs+D+np.sqrt((Zs+D)**2+R0**2))/(Zs-D+np.sqrt((Zs-D)**2+R0**2)));
    q2 = RHOs**2/4*((Zs+D)/((Zs+D)**2+R0**2)**1.5 - (Zs-D)/((Zs-D)**2+R0**2)**1.5);
    Bz = muI_2*(q1+q2);
    return Brho, Bz;

def LargeApprox(RHOs,Zs,R0,D,muI_2):
    """Compute the magnetic field cause by a magnetodisc (large rho approximation)
    Ref: Edwards, T. M., Bunce, E. J., & Cowley, S. W. H. (2001). 
         A note on the vector potential of Connerney et al.'s model of the 
         equatorial current sheet in Jupiter's magnetosphere. 
         Planetary and Space Science, 49, 1115-1123. https://doi.org/10.1016/S0032-0633(00)00164-1

    Args:
        RHOs,Zs: (numpy array (N,)): Magnetic Dipote coordinates (Jovian radii)
        R0: Disc inner radius (Jovian radii)
        D: Half thickness (Jovian radii)
        muI_2: Current constant (nT)

    Returns:
        Brho, Bz (numpy array (N,)): Magnetic field components and magnitude
    """
    p1 = 1/RHOs*(np.sqrt((Zs-D)**2+RHOs**2)-np.sqrt((Zs+D)**2+RHOs**2));
    p2 = RHOs/4*R0**2*(1/((Zs+D)**2+RHOs**2)**1.5 - 1/((Zs-D)**2+RHOs**2)**1.5);
    p3 = 2/RHOs*np.sign(Zs)*np.minimum(np.abs(Zs),D);
    Brho = muI_2*(p1+p2+p3);
    q1 = np.log((Zs+D+np.sqrt((Zs+D)**2+RHOs**2))/(Zs-D+np.sqrt((Zs-D)**2+RHOs**2)));
    q2 = R0**2/4*((Zs+D)/((Zs+D)**2+RHOs**2)**1.5 - (Zs-D)/((Zs-D)**2+RHOs**2)**1.5);
    Bz = muI_2*(q1+q2);
    return Brho, Bz;