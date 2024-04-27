##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Table showing the RMS errors of `Spherical Harmonic Models` 
# computed at different subset of ***Juno Observation Orbits***
##################################################################
# %%
# # ! Setup
import numpy as np;
from LW_DataReadWrite import *;
from LW_CoordinateTransformation import *;
from LW_SH_Mag import *;
# %%
# # ! SH Model Estimations of [Bx,By,Bz] at <Multiple Different Obs Dataset> using different <nmax>
# TODO ******************** Parameters ******************** #
cstRj = 71492;        # Jupiter radius in km
GS2nT = 1e5;          # Gauss to nt
cutRJ = 2.0;          # Data for statistics
# TODO ********************** end ************************* #
######## Test Juno OBS dataset
FILEOBSs = ['input/Connerney_PJ01_33_4.0Rj.txt',      # <Connerney Obs dataset PJ01-33>
            'input/Juno_PJ01_33_4.0Rj.txt',           # <Our Obs dataset PJ01-33>
            'input/RawJuno_PJ01_33_4.0Rj.txt',        # <Raw Obs dataset PJ01-33>
            'input/Juno_PJ01_50_4.0Rj.txt'];          # <Our Obs dataset PJ01-50>  
OBSNAMEs = ['CnyObs33',
            'OurObs33',
            'RawObs33',
            'OurObs50'];
######## Interior SH models
FILESHs_INT = ['input/JRM33_I30.txt',
               'input/Bloxham_I32.txt'];
NMAXs = [2,32];    # Choose the expansion degree n
######## Exterior MagnetoDisc models
MDa = {'R0':7.8, 'R1':51.4, 'D':3.6, 'muI_2':139.6, 'md_lon':204.2, 'md_lat':9.3};
# MDb = {'R0':5.0, 'R1':50.0, 'D':2.5, 'muI_2':225.0, 'md_lon':204.2, 'md_lat':9.3};
MDb = {'R0':5.0, 'R1':50.0, 'D':2.5, 'muI_2':225.0, 'md_lon':196.61, 'md_lat':10.31};
MAGDISCs  = [MDa,MDb];
######## Computation
SHNAMEs = ['JRM33_I30MDa','Bloxham_I32MDb'];
pkNMAXs = [[1,2],[18,32]];
for iModel in range(len(FILESHs_INT)):
     nmax =NMAXs[iModel];
     shName = SHNAMEs[iModel];
     fileSH_int = FILESHs_INT[iModel];
     MagDisc = MAGDISCs[iModel];
     print('\n*********************\n',flush=True);
     print('RMS of %s Model evaluated at Juno <OBS>:'%(shName),flush=True);
     for n in range(nmax):
          if ((n+1) in pkNMAXs[iModel]):
               print('\033[41mn = %2d; \033[0m'%(n+1),end='',flush=True);
          else:
               print('n = %2d; '%(n+1),end='',flush=True);
          for iFILEOBS in range(len(FILEOBSs)):
               fileObs = FILEOBSs[iFILEOBS];
               nObs,PJ,Year,DD,xObs,yObs,zObs,BxObs,ByObs,BzObs = LoadObsFile(fileObs,showinfo=False);
               xObs=xObs/cstRj; yObs=yObs/cstRj; zObs=zObs/cstRj;          # Distance in RJ
               RObs=np.sqrt(xObs**2+yObs**2+zObs**2); 
               pkInd = (RObs<=cutRJ);
               xObs = xObs[pkInd].reshape(-1,1); 
               yObs = yObs[pkInd].reshape(-1,1); 
               zObs = zObs[pkInd].reshape(-1,1);
               BxObs = BxObs[pkInd].reshape(-1,1); 
               ByObs = ByObs[pkInd].reshape(-1,1); 
               BzObs = BzObs[pkInd].reshape(-1,1);
               BxObs=GS2nT*BxObs; ByObs=GS2nT*ByObs; BzObs=GS2nT*BzObs;    # Magnetic in nT
               refBNorm = np.sqrt(BxObs**2+ByObs**2+BzObs**2);
               if ((n+1) in pkNMAXs[iModel]):
                    print('\033[41m%9s, \033[0m'%(OBSNAMEs[iFILEOBS]),end='',flush=True);
               else:
                    print('%9s, '%(OBSNAMEs[iFILEOBS]),end='',flush=True);
               estBx_I,estBy_I,estBz_I,_ = SHS_Bxyz(fileSH_int,n+1,1.0,xObs,yObs,zObs,showinfo=False);
               estBx_I=GS2nT*estBx_I; estBy_I=GS2nT*estBy_I; estBz_I=GS2nT*estBz_I;    # Magnetic in nT
               mdXs, mdYs, mdZs = ecef2MD(xObs, yObs, zObs, MagDisc['md_lon'], MagDisc['md_lat']);
               mdBx, mdBy, mdBz, mdBNorm  = \
                    MagnetoDisc(mdXs, mdYs, mdZs, MagDisc['R0'], MagDisc['R1'], MagDisc['D'], MagDisc['muI_2']);
               estBx_E,estBy_E,estBz_E,_ = MD2ecef_v(mdBx, mdBy, mdBz, MagDisc['md_lon'], MagDisc['md_lat']);
               estBx = estBx_I + estBx_E;
               estBy = estBy_I + estBy_E;
               estBz = estBz_I + estBz_E;
               estBNorm = np.sqrt(estBx**2+estBy**2+estBz**2); 
               err_dB = estBNorm - refBNorm;   # Delta |B|
               err_DB = np.sqrt((estBx-BxObs)**2+(estBy-ByObs)**2+(estBz-BzObs)**2);   # |Delta B|
               err_dBxyz = np.concatenate((estBx-BxObs,estBy-ByObs,estBz-BzObs),axis=0);
               # rms = np.sqrt(np.mean(err_dB**2));
               # rms = np.sqrt(np.mean(err_DB**2));
               rms = np.sqrt(np.mean(err_dBxyz**2));
               if ((n+1) in pkNMAXs[iModel]):
                    print('\033[41mrms = %8.1f nT; \033[0m'%(rms),end='',flush=True);
               else:
                    print('rms = %8.1f nT; '%(rms),end='',flush=True);
          print('',flush=True); # <br>