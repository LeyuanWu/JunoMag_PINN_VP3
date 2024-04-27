##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Predict magnetic vector fields at ***Multiple*** $R_j$ 
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
pj1=1; pj2=50;            # orbit 1-33 | orbit 1-50 | others
cutRType=2;               # 1: 2.5Rj; 2: 4.0Rj; 3: 7.0Rj
nLayer=6; nNeuron=40;
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
# # ! Estimations of [Bn,Be,Bd] & |B| at <Rj>s and Save to ascii file
# TODO ******************** Parameters ******************** #
# RJs=[1.00,1.05,1.10,1.15,1.20];    # For PINN traning above 1.00Rj
RJs=[1.25,1.30,1.35,1.40,1.45,1.50,1.60,1.70,1.80,1.90,2.00,2.50,3.00,3.50,4.00];
LambdaP=np.linspace(np.deg2rad(0.5),2*np.pi-np.deg2rad(0.5),360);
ThetaP=np.linspace(np.deg2rad(0.5),np.pi-np.deg2rad(0.5),180); # ! Co-latitude
# TODO ********************** end ************************* #
######## Mapping Juno Mag
LambdaP2d,ThetaP2d = np.meshgrid(LambdaP,ThetaP);
LambdaPs=LambdaP2d.reshape(-1,1); ThetaPs=ThetaP2d.reshape(-1,1);
nRJ=len(RJs);
for iRJ in range(nRJ):
    rj=RJs[iRJ];
    curR=rj*GetConstant('Rj');
    xPred,yPred,zPred=ell2ecef(LambdaPs,np.pi/2-ThetaPs,0,curR,0);
    bxPred, byPred, bzPred = model.predict_curl_A(xPred,yPred,zPred);
    Bn,Be,Bd=ecef2ned_v(xPred,yPred,zPred,bxPred,byPred,bzPred,curR,0);
    BNorm=np.sqrt(bxPred**2+byPred**2+bzPred**2); 
    ######## Write B estimations to file
    fileEstBnedRjs=fNameHead + '_EstBned_%.2fRj'%(rj)+'.txt';
    SaveBnedFile(fileEstBnedRjs,LambdaPs,ThetaPs,Bn,Be,Bd,BNorm);