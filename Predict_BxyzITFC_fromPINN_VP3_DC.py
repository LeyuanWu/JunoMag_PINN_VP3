##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Predict magnetic vector fields at ***User-Defined*** locations 
# using `PINN models` and write to ascii file
##################################################################
# %%
# # ! Setup
import numpy as np;
from LW_PINN_VP3_JUNO import *;
from LW_CoordinateTransformation import *;
from LW_DataReadWrite import *;
# %%
# # ! Obs & Col & NN Hyper-Parameters 
# TODO ******************** Parameters ******************** #
pj1=1; pj2=33;            # orbit 1-33 | orbit 1-50 | others
cutRType=2;               # 1: 2.5Rj; 2: 4.0Rj; 3: 7.0Rj
nLayer=6; nNeuron=40;
actiFun='swish';          # 'tanh' 'gelu' 'siren' 'swish' 'relu' 'sigmoid'
opti='Adam';              # 'Adam'
DW=1;                     # Dynamic Weights 1/0 for On/Off
k=1; c=0; n0=3000; dn=600;   # 2023WuCX
nEpo=12000;
BS=10000; 
rdSeed=67890;
# TODO ********************** end ************************* #
fileObs='input/JUNO_PINN_VP3_PJ%02d_%02d_4.0Rj_NN06_040_swish_Adam_DW1_'%(pj1,pj2)\
     +'RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_Itfc.txt';
fileCol='input/Collocation_Random_minR0.80_maxR1.00_40000.txt';
fNameHead, model = BuildPINN_DC(pj1,pj2,cutRType,fileObs,fileCol,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
# %%
# # ! Using Loss Function to find the best NN model with minimum loss in the last 10 iteration
######## Read Loss Terms
fileLossVars='output/'+fNameHead+'_LossVars.txt';
dataLossVars=np.loadtxt(fileLossVars);
true_loss=dataLossVars[:,2:3];
print('Finish read data from file: %s'%(fileLossVars),flush=True);
pkEpo=np.argmin(true_loss[nEpo-10:nEpo]) + nEpo-10+1;
print('Picked Epoch: %d'%(pkEpo),flush=True);
model.saver.restore(model.sess,save_path='./'+fNameHead+'/ckpt-%d'%(pkEpo));
# %%
# # ! Estimations of [Bx,By,Bz] at 80000 random points on <r=1.00Rj>
fileItfc='input/Interface_Random_R1.00_80000.txt';
nItfc,xItfc,yItfc,zItfc=LoadItfcData(fileItfc);
bx_obs_est, by_obs_est, bz_obs_est = model.predict_curl_A(xItfc,yItfc,zItfc);
fileEstBxyzItfc = fNameHead + '_EstBxyz_Itfc.txt';
SaveItfcData(fileEstBxyzItfc,xItfc,yItfc,zItfc,bx_obs_est,by_obs_est,bz_obs_est);