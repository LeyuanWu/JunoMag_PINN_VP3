##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Predict magnetic vector fields at ***Juno Observation Locations*** 
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
pj1=1; pj2=50;            # orbit 1-50 | orbit 1-24 | orbit 25-50
cutRType=2;               # 1: 2.5Rj; 2: 4.0Rj; 3: 7.0Rj
nLayer=8; nNeuron=40;
actiFun='swish';          # 'tanh' 'gelu' 'siren' 'swish' 'relu' 'sigmoid'
opti='Adam';              # 'Adam'
DW=1;                     # Dynamic Weights 1/0 for On/Off
k=1; c=0; n0=3000; dn=600;   # 2023WuCX
nEpo=12000;
BS=10000; 
rdSeed=12345;
# TODO ********************** end ************************* #
fileCol='input/Collocation_Random_minR1.00_maxR4.00_300000.txt';
fNameHead, model = BuildPINN(pj1,pj2,cutRType,fileCol,nLayer,nNeuron,\
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
# # ! Estimations of [Bx,By,Bz] at <Multiple Different Obs Dataset> and Save to ascii file
FILEOBSs = ['input/Connerney_PJ01_33_4.0Rj.txt',      # <Connerney Obs dataset PJ01-33>
            'input/Juno_PJ01_33_4.0Rj.txt',           # <Our Obs dataset PJ01-33>
            'input/RawJuno_PJ01_33_4.0Rj.txt',        # <Raw Juno Obs dataset PJ01-33>
            'input/Juno_PJ01_50_4.0Rj.txt'];          # <Our Obs dataset PJ01-50>  
Suffix = ['_EstBxyz_ConnerneyObs33.txt',
          '_EstBxyz_OurObs33.txt',
          '_EstBxyz_RawObs33.txt',
          '_EstBxyz_OurObs50.txt'];
for iFILEOBS in range(len(FILEOBSs)):
     fileObs = FILEOBSs[iFILEOBS];
     nObs,PJ,Year,DD,xObs,yObs,zObs,bxObs,byObs,bzObs = LoadObsFile(fileObs,showinfo=True);
     bx_obs_est, by_obs_est, bz_obs_est = model.predict_curl_A(xObs,yObs,zObs);
     fileEstBxyzOBS = fNameHead + Suffix[iFILEOBS];
     SaveObsFile(fileEstBxyzOBS,PJ,Year,DD,xObs,yObs,zObs,bx_obs_est,by_obs_est,bz_obs_est,showinfo=True);