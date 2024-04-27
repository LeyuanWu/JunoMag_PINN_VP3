##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Predict magnetic vector fields at ***Multiple*** $R_j$ 
# using `Spherical Harmonic Models` and write to ascii file
##################################################################
# %%
# # ! Setup
import numpy as np;
from LW_SH_Mag import *;
from LW_DataReadWrite import *;
# %%
# # ! Global parameter
# TODO ******************** Parameters ******************** #
cstRj=71492;      # Jupiter radius in km
# TODO ********************** end ************************* #
# %%
# # ! SH Model Estimations of [Bn,Be,Bd] & |B| at <Rj>s (Connerney et al., 2022)
# TODO ******************** Parameters ******************** #
nmax = 30;
fileSH = 'input/JRM33_I30.txt';
fNameHead = 'JRM33_I30_nmax%d'%(nmax);
RJs = [1.00,0.95,0.90,0.85,0.80];   # For PINN Downward Continuation
LambdaP=np.linspace(np.deg2rad(0.5),2*np.pi-np.deg2rad(0.5),360);
ThetaP=np.linspace(np.deg2rad(0.5),np.pi-np.deg2rad(0.5),180); # ! Co-latitude
# TODO ********************** end ************************* #
LambdaP2d,ThetaP2d = np.meshgrid(LambdaP,ThetaP);
LambdaPs=LambdaP2d.reshape(-1,1); ThetaPs=ThetaP2d.reshape(-1,1);
nRJ=len(RJs);
for iRJ in range(nRJ):
    rj=RJs[iRJ];
    curR=rj*cstRj;
    RPs = curR*np.ones(ThetaPs.shape);
    Bn,Be,Bd = SHS_Bned(fileSH,nmax,cstRj,RPs,ThetaPs,LambdaPs,showinfo=False);
    BNorm=np.sqrt(Bn**2+Be**2+Bd**2); 
    ######## Write B estimations to file
    fileEstBnedRjs=fNameHead + '_EstBned_%.2fRj'%(rj)+'.txt';
    SaveBnedFile(fileEstBnedRjs,LambdaPs,ThetaPs,Bn,Be,Bd,BNorm);
# %%
# # ! SH Model Estimations of [Bn,Be,Bd] & |B| at <Rj>s (Bloxham et al., 2022)
# TODO ******************** Parameters ******************** #
nmax = 18;
fileSH = 'input/Bloxham_I32.txt';
fNameHead = 'Bloxham_I32_nmax%d'%(nmax);
RJs = [1.00,0.95,0.90,0.85,0.80];   # For PINN Downward Continuation
LambdaP=np.linspace(np.deg2rad(0.5),2*np.pi-np.deg2rad(0.5),360);
ThetaP=np.linspace(np.deg2rad(0.5),np.pi-np.deg2rad(0.5),180); # ! Co-latitude
# TODO ********************** end ************************* #
LambdaP2d,ThetaP2d = np.meshgrid(LambdaP,ThetaP);
LambdaPs=LambdaP2d.reshape(-1,1); ThetaPs=ThetaP2d.reshape(-1,1);
nRJ=len(RJs);
for iRJ in range(nRJ):
    rj=RJs[iRJ];
    curR=rj*cstRj;
    RPs = curR*np.ones(ThetaPs.shape);
    Bn,Be,Bd = SHS_Bned(fileSH,nmax,cstRj,RPs,ThetaPs,LambdaPs,showinfo=False);
    BNorm=np.sqrt(Bn**2+Be**2+Bd**2); 
    ######## Write B estimations to file
    fileEstBnedRjs=fNameHead + '_EstBned_%.2fRj'%(rj)+'.txt';
    SaveBnedFile(fileEstBnedRjs,LambdaPs,ThetaPs,Bn,Be,Bd,BNorm);