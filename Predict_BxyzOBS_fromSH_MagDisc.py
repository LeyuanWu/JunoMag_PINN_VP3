##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Predict magnetic vector fields at ***Juno Observation Locations*** 
# & ***Multiple*** $R_j$ using `Spherical Harmonic Models` 
# and write to ascii file
##################################################################
# %%
# # ! Setup
import numpy as np;
from LW_SH_Mag import *;
from LW_DataReadWrite import *;
# %%
# # ! SH Model Estimations of [Bx,By,Bz] at <Multiple Different Obs Dataset> and Save to ascii file
FILEOBSs = ['input/Connerney_PJ01_33_4.0Rj.txt',      # <Connerney Obs dataset PJ01-33>
            'input/Juno_PJ01_33_4.0Rj.txt',           # <Our Obs dataset PJ01-33>
            'input/RawJuno_PJ01_33_4.0Rj.txt',        # <Raw Juno Obs dataset PJ01-33>
            'input/Juno_PJ01_50_4.0Rj.txt'];          # <Our Obs dataset PJ01-50>  
Suffix = ['_EstBxyz_ConnerneyObs33.txt',
          '_EstBxyz_OurObs33.txt',
          '_EstBxyz_RawObs33.txt',
          '_EstBxyz_OurObs50.txt'];
# TODO ******************** Parameters ******************** #
cstRj=71492;        # Jupiter radius in km
GS2nT=1e5;          # Gauss to nt
nT2GS = 1e-5;       # nT to Gauss
# TODO ********************** end ************************* #
######## Interior SH models
FILESHs_INT = ['input/JRM33_I30.txt',
               'input/Bloxham_I32.txt'];
NMAXs = [30,32];    # Choose the expansion degree n
######## Exterior MagnetoDisc models
MDa = {'R0':7.8, 'R1':51.4, 'D':3.6, 'muI_2':139.6, 'md_lon':204.2, 'md_lat':9.3};
MDb = {'R0':5.0, 'R1':50.0, 'D':2.5, 'muI_2':225.0, 'md_lon':204.2, 'md_lat':9.3};
MAGDISCs  = [MDa,MDb];
######## Computation
SHNAMEs = ['JRM33_I30MDa','Bloxham_I32MDb'];
for iFILEOBS in range(len(FILEOBSs)):
     fileObs = FILEOBSs[iFILEOBS];
     nObs,PJ,Year,DD,xObs,yObs,zObs,BxObs,ByObs,BzObs = LoadObsFile(fileObs,showinfo=False);
     xObs=xObs/cstRj; yObs=yObs/cstRj; zObs=zObs/cstRj;          # Distance in RJ
     BxObs=GS2nT*BxObs; ByObs=GS2nT*ByObs; BzObs=GS2nT*BzObs;    # Magnetic in nT
     for iModel in range(len(FILESHs_INT)):
          fileSH_int = FILESHs_INT[iModel];
          nmax = NMAXs[iModel];
          MagDisc = MAGDISCs[iModel];
          fNameHead = SHNAMEs[iModel] + '_nmax%d'%(nmax);
          estBx_I,estBy_I,estBz_I,_ = SHS_Bxyz(fileSH_int,nmax,1.0,xObs,yObs,zObs,showinfo=False);
          estBx_I=GS2nT*estBx_I; estBy_I=GS2nT*estBy_I; estBz_I=GS2nT*estBz_I;    # Magnetic in nT
          mdXs, mdYs, mdZs = ecef2MD(xObs, yObs, zObs, MagDisc['md_lon'], MagDisc['md_lat']);
          mdBx, mdBy, mdBz, mdBNorm  = \
               MagnetoDisc(mdXs, mdYs, mdZs, MagDisc['R0'], MagDisc['R1'], MagDisc['D'], MagDisc['muI_2']);
          estBx_E,estBy_E,estBz_E,_ = MD2ecef_v(mdBx, mdBy, mdBz, MagDisc['md_lon'], MagDisc['md_lat']);
          estBx = estBx_I + estBx_E;
          estBy = estBy_I + estBy_E;
          estBz = estBz_I + estBz_E;
          estBx = nT2GS*estBx; estBy = nT2GS*estBy; estBz = nT2GS*estBz;    # Magnetic in Gauss
          #### Write to file
          fileEstBxyzOBS = fNameHead + Suffix[iFILEOBS];
          SaveObsFile(fileEstBxyzOBS,PJ,Year,DD,cstRj*xObs,cstRj*yObs,cstRj*zObs,estBx,estBy,estBz,showinfo=True);