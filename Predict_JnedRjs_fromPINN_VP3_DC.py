##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Predict Currents J at ***Multiple*** $R_j$ 
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
# # ! Estimations of [Jn,Je,Jd] & |J| at <Rj>s and Save to ascii file
# TODO ******************** Parameters ******************** #
RJs=[0.80,0.90];    # For PINN Downward Continuation
LambdaP=np.linspace(np.deg2rad(0.5),2*np.pi-np.deg2rad(0.5),360);
ThetaP=np.linspace(np.deg2rad(0.5),np.pi-np.deg2rad(0.5),180); # ! Co-latitude
scaleUnit = model.cordScale / (4*np.pi); # Gauss/km to A/m^2
# TODO ********************** end ************************* #
######## Mapping Juno Mag
LambdaP2d,ThetaP2d = np.meshgrid(LambdaP,ThetaP);
LambdaPs=LambdaP2d.reshape(-1,1); ThetaPs=ThetaP2d.reshape(-1,1);
nRJ=len(RJs);
for iRJ in range(nRJ):
    rj=RJs[iRJ];
    curR=rj*GetConstant('Rj');
    xPred,yPred,zPred=ell2ecef(LambdaPs,np.pi/2-ThetaPs,0,curR,0);
    jxPred, jyPred, jzPred = model.predict_curl_curl_A(xPred,yPred,zPred);
    Jn,Je,Jd=ecef2ned_v(xPred,yPred,zPred,jxPred,jyPred,jzPred,curR,0);
    Jn = scaleUnit*Jn; Je = scaleUnit*Je; Jd = scaleUnit*Jd;
    JNorm = scaleUnit*np.sqrt(jxPred**2+jyPred**2+jzPred**2); 
    ######## Write J estimations to file
    fileEstJnedRjs=fNameHead + '_EstJned_%.2fRj'%(rj)+'.txt';
    SaveJnedFile(fileEstJnedRjs,LambdaPs,ThetaPs,Jn,Je,Jd,JNorm);