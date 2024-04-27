# %% # ! Import
import numpy as np;

def LoadObsFile(fileObs,showinfo=True):   
    # ***************** Format ******************
    # PJNUM Year Decimal-Day X Y Z Bx By Bz
    # *******************************************
    Rj=71492; # Jupiter radius in km
    dataObs=np.loadtxt(fileObs,skiprows=1);
    PJ=dataObs[:,0:1]; Year=dataObs[:,1:2]; DD=dataObs[:,2:3]; 
    xObs=dataObs[:,3:4]; yObs=dataObs[:,4:5]; zObs=dataObs[:,5:6]; 
    bxObs=dataObs[:,6:7]; byObs=dataObs[:,7:8]; bzObs=dataObs[:,8:9];
    RObs=np.sqrt(xObs**2+yObs**2+zObs**2); 
    RObsRj=RObs/Rj;
    bNorm=np.sqrt(bxObs**2+byObs**2+bzObs**2); 
    nObs=xObs.size; 
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('Number of <Obs>: %d'%(nObs),flush=True);
        print('PJ in [%d,%d];'%(PJ.min(),PJ.max()),flush=True);
        print('Year in [%d,%d];'%(Year.min(),Year.max()),flush=True);
        print('x in [%.4f,%.4f] km;'%(xObs.min(),xObs.max()),flush=True);
        print('y in [%.4f,%.4f] km;'%(yObs.min(),yObs.max()),flush=True);
        print('z in [%.4f,%.4f] km;'%(zObs.min(),zObs.max()),flush=True);
        print('RObs in [%.4f,%.4f] km;'%(RObs.min(),RObs.max()),flush=True);
        print('RObs in [%.3f,%.3f] Rj;'%(RObsRj.min(),RObsRj.max()),flush=True);
        print('bx in [%.9f,%.6f] Gauss;'%(bxObs.min(),bxObs.max()),flush=True);
        print('by in [%.6f,%.6f] Gauss;'%(byObs.min(),byObs.max()),flush=True);
        print('bz in [%.6f,%.6f] Gauss;'%(bzObs.min(),bzObs.max()),flush=True);
        print('bNorm in [%.6f,%.6f] Gauss;'%(bNorm.min(),bNorm.max()),flush=True);
        print('\n*****************************************\n',flush=True);
        print('Distribution of <Obs>: %d'%(nObs),flush=True);
        Rbins=np.concatenate((np.linspace(0.99,1.10,12)[:,None],\
                            np.linspace(1.2,2.0,9)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RObsRj>=r1,RObsRj<r2));
            print('Number of Obs in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
        print('\n*****************************************\n',flush=True);
    return nObs,PJ,Year,DD,xObs,yObs,zObs,bxObs,byObs,bzObs;

def SaveObsFile(fileObs,PJ,Year,DD,xObs,yObs,zObs,bx_obs_est,by_obs_est,bz_obs_est,showinfo=True):
    # ***************** Format ******************
    # PJNUM Year Decimal-Day X Y Z Bx By Bz
    # *******************************************
    Rj=71492; # Jupiter radius in km
    fid = open(fileObs, 'w');
    wrtArray=np.hstack((PJ,Year,DD,xObs,yObs,zObs,bx_obs_est,by_obs_est,bz_obs_est));
    wrtFormat=['%02d','%d','%13.9f',\
        '%12.3f','%12.3f','%12.3f',\
        '%12.7f','%12.7f','%12.7f'];
    colHead='%2s %4s %13s '%('PJ','Year','DD')\
        +'%12s %12s %12s %12s %12s %12s'%('x','y','z','Est_Bx','Est_By','Est_Bz');
    np.savetxt(fid,wrtArray,fmt=wrtFormat,header=colHead);
    fid.close();
    print('\n*****************************************\n',flush=True);
    print('Saving estimation of [Bx,By,Bz] at <Obs> to file: %s'%(fileObs),flush=True);
    RObs=np.sqrt(xObs**2+yObs**2+zObs**2); 
    RObsRj=RObs/Rj;
    bNorm_obs_est=np.sqrt(bx_obs_est**2+by_obs_est**2+bz_obs_est**2); 
    nObs=xObs.size; 
    if showinfo:
        print('Number of <Obs>: %d'%(nObs),flush=True);
        print('PJ in [%d,%d];'%(PJ.min(),PJ.max()),flush=True);
        print('Year in [%d,%d];'%(Year.min(),Year.max()),flush=True);
        print('RObs in [%.3f,%.3f] Rj;'%(RObsRj.min(),RObsRj.max()),flush=True);
        print('bNorm in [%.6f,%.6f] Gauss;'%(bNorm_obs_est.min(),bNorm_obs_est.max()),flush=True);
        print('\n \n',flush=True);
        print('Distribution of <Obs>: %d'%(nObs),flush=True);
        Rbins=np.concatenate((np.linspace(0.99,1.10,12)[:,None],\
                            np.linspace(1.2,2.0,9)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RObsRj>=r1,RObsRj<r2));
            print('Number of Obs in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
    print('\n*****************************************\n',flush=True);
    return None;

def SaveBnedFile(fileBned,LambdaPs,ThetaPs,Bn,Be,Bd,BNorm,showinfo=True):
    # ***************** Format ******************
    # Lon Lat Bn Be -Bd |B|
    # *******************************************
    fid = open(fileBned, 'w');
    wrtArray=np.hstack((np.rad2deg(LambdaPs),np.rad2deg(np.pi/2-ThetaPs),\
        Bn,Be,-Bd,BNorm));
    wrtFormat=['%9.3f','%9.3f','%12.7f','%12.7f','%12.7f','%12.7f'];
    colHead='%9s %9s %12s %12s %12s %12s'%('Lon','Lat','Est_Bn','Est_Be','Est_-Bd','Est_|B|');
    np.savetxt(fid,wrtArray,fmt=wrtFormat,header=colHead);
    fid.close();
    print('\n*****************************************\n',flush=True);
    print('Saving SH estimation of [Bn,Be,-Bd] & |B| to file: %s'%(fileBned),flush=True);
    if showinfo:
        print('bNorm in [%.6f,%.6f] Gauss;'%(BNorm.min(),BNorm.max()),flush=True);
    print('\n*****************************************\n',flush=True);
    return None;

def SaveJnedFile(fileJned,LambdaPs,ThetaPs,Jn,Je,Jd,JNorm,showinfo=True):
    # ***************** Format ******************
    # Lon Lat Jn Je -Jd |J|
    # *******************************************
    fid = open(fileJned, 'w');
    wrtArray=np.hstack((np.rad2deg(LambdaPs),np.rad2deg(np.pi/2-ThetaPs),\
        Jn,Je,-Jd,JNorm));
    wrtFormat=['%9.3f','%9.3f','%12.2e','%12.2e','%12.2e','%12.2e'];
    colHead='%9s %9s %12s %12s %12s %12s'%('Lon','Lat','Est_Jn','Est_Je','Est_-Jd','Est_|J|');
    np.savetxt(fid,wrtArray,fmt=wrtFormat,header=colHead);
    fid.close();
    print('\n*****************************************\n',flush=True);
    print('Saving SH estimation of [Jn,Je,-Jd] & |J| to file: %s'%(fileJned),flush=True);
    if showinfo:
        print('JNorm in [%.2e,%.2e] A/m^2;'%(JNorm.min(),JNorm.max()),flush=True);
    print('\n*****************************************\n',flush=True);
    return None;

def LoadItfcData(fileItfc,showinfo=True):
    # ******** Format *********
    # X Y Z 
    # *************************
    Rj=71492; # Jupiter radius in km
    dataItfc = np.loadtxt(fileItfc,skiprows=1);
    xItfc=dataItfc[:,0:1]; yItfc=dataItfc[:,1:2]; zItfc=dataItfc[:,2:3];
    RItfc=np.sqrt(xItfc**2+yItfc**2+zItfc**2);
    RItfcRj=RItfc/Rj;
    nItfc=xItfc.size; 
    if showinfo:
        print('\n*****************************************\n',flush=True);
        print('Number of <Interface Points>: %d'%(nItfc),flush=True);
        print('RItfc in: [%.4f,%.4f] km;'%(RItfc.min(),RItfc.max()),flush=True);
        print('RItfc in: [%.3f,%.3f] Rj;'%(RItfcRj.min(),RItfcRj.max()),flush=True);
        print('Distribution of <Interace Points>: %d'%(nItfc),flush=True);
        Rbins=np.concatenate((np.linspace(0.80,1.00,21)[:,None],\
                            np.linspace(1.1,2.0,10)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RItfcRj>=r1,RItfcRj<r2));
            print('Number of Interace Points in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
        print('\n*****************************************\n',flush=True);
    return nItfc,xItfc,yItfc,zItfc;

def SaveItfcData(fileItfc,xItfc,yItfc,zItfc,bx_obs_est,by_obs_est,bz_obs_est,showinfo=True):
    # ******** Format *********
    # X Y Z Bx By Bz
    # *************************
    Rj=71492; # Jupiter radius in km
    fid = open(fileItfc, 'w');
    wrtArray=np.hstack((xItfc,yItfc,zItfc,bx_obs_est,by_obs_est,bz_obs_est));
    wrtFormat=['%12.3f','%12.3f','%12.3f','%12.7f','%12.7f','%12.7f'];
    colHead='%12s %12s %12s %12s %12s %12s'%('x','y','z','Est_Bx','Est_By','Est_Bz');
    np.savetxt(fid,wrtArray,fmt=wrtFormat,header=colHead);
    fid.close();
    print('\n*****************************************\n',flush=True);
    print('Saving estimation of [Bx,By,Bz] at <User-Defined> to file: %s'%(fileItfc),flush=True);
    RItfc=np.sqrt(xItfc**2+yItfc**2+zItfc**2); 
    RItfcRj=RItfc/Rj;
    bNorm_obs_est=np.sqrt(bx_obs_est**2+by_obs_est**2+bz_obs_est**2); 
    nItfc=xItfc.size; 
    if showinfo:
        print('Number of <Itfc>: %d'%(nItfc),flush=True);
        print('RItfc in [%.3f,%.3f] Rj;'%(RItfcRj.min(),RItfcRj.max()),flush=True);
        print('bNorm in [%.6f,%.6f] Gauss;'%(bNorm_obs_est.min(),bNorm_obs_est.max()),flush=True);
        print('\n \n',flush=True);
        print('Distribution of <Itfc>: %d'%(nItfc),flush=True);
        Rbins=np.concatenate((np.linspace(0.99,1.10,12)[:,None],\
                            np.linspace(1.2,2.0,9)[:,None],\
                            np.linspace(2.5,4.0,4)[:,None],\
                            np.linspace(5.0,8.0,4)[:,None]));
        nRbin=Rbins.size;
        for i in range(nRbin-1):
            r1=Rbins[i]; r2=Rbins[i+1];
            n12=np.count_nonzero(np.logical_and(RItfcRj>=r1,RItfcRj<r2));
            print('Number of Itfc in [%.2f,%.2f) Rj: %d'%(r1,r2,n12),flush=True);
    print('\n*****************************************\n',flush=True);
    return None;