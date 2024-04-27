##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Least-Square Estimation of Spherical Harmonic Coeeficients of 
# PINN predicted magnetic fields at certain RJ
##################################################################
# %%
# # ! Setup
import numpy as np;
from scipy import linalg;
from scipy import stats;
from LW_SH_Mag import *;
# %%
# # ! Data
dataFiles=[
    'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBned_1.00Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.95Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.90Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.85Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.80Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBned_1.00Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.95Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.90Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.85Rj.txt',\
    'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed67890_EstBned_0.80Rj.txt'
    ]
shcMODELs = [
             'NN06_040_PINN33e_1.00Rj_I35',\
             'NN06_040_PINN33i_0.95Rj_I35',\
             'NN06_040_PINN33i_0.90Rj_I35',\
             'NN06_040_PINN33i_0.85Rj_I35',\
             'NN06_040_PINN33i_0.80Rj_I35',\
             'NN06_040_PINN50e_1.00Rj_I35',\
             'NN06_040_PINN50i_0.95Rj_I35',\
             'NN06_040_PINN50i_0.90Rj_I35',\
             'NN06_040_PINN50i_0.85Rj_I35',\
             'NN06_040_PINN50i_0.80Rj_I35'
             ];
# %%
# # ! Least-Square Estimation of SHCs
# TODO ******************** Parameters ******************** #
cstRJ = 71492;      # Jupiter radius in km
rj = 1.0;
nmax_PINN = 35;      
LambdaP=np.linspace(np.deg2rad(0.5),2*np.pi-np.deg2rad(0.5),360);
ThetaP=np.linspace(np.deg2rad(0.5),np.pi-np.deg2rad(0.5),180); # ! Co-latitude
# TODO ********************** end ************************* #
LambdaP2d,ThetaP2d = np.meshgrid(LambdaP,ThetaP);
LambdaPs = LambdaP2d.reshape(-1,1); ThetaPs = ThetaP2d.reshape(-1,1);
curR=rj*cstRJ;
RPs = curR*np.ones(ThetaPs.shape);
Gn, Ge, Gd = design_Gauss(nmax_PINN,cstRJ,RPs,ThetaPs,LambdaPs,showinfo=True);
nMODEL = len(shcMODELs);
for iModel in range(nMODEL):
    Bd = (-1)* np.loadtxt(dataFiles[iModel],skiprows=1,usecols=4);
    ghSHCs = ((linalg.inv ( Gd.T @ Gd )) @ Gd.T @ Bd).reshape(-1,);
    shcFileName = shcMODELs[iModel] + '.txt';
    Format_SHCsFile(ghSHCs,nmax_PINN,shcFileName,internal=True);