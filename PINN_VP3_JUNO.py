###############################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
###############################################
# %% # ! Setup
import tensorflow as tf;
import numpy as np;
from LW_PINN_VP3_JUNO import *;
# %% # ! Model setting
# TODO ******************** Parameters ******************** #
pj1=1; pj2=33; # orbit 1-50 | orbit 1-24 | orbit 25-50 | others
cutRType=2; # 1: 2.5Rj; 2: 4.0Rj; 3: 7.0Rj
# TODO ********************** end ************************* #
# %% # ! NN Hyper-Parameters 
# TODO ******************** Parameters ******************** #
nLayer=8; nNeuron=40;
actiFun='swish';              # 'tanh' 'gelu' 'siren' 'swish' 'relu' 'sigmoid'
opti='Adam';                  # 'Adam'
DW=1;                         # Dynamic Weights 1/0 for On/Off
k=1; c=0; n0=3000; dn=600;    # 2023WuCX
nEpo=12000; nBatPerEpo=30;
BS=10000; 
rdSeed=12345;
# TODO ********************** end ************************* #
######## Build <File Name Head>
fNameHead=BuildFileNameHead(pj1,pj2,cutRType,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
print('\n*****************************************\n',flush=True);
print('<File Name Head>: %s'%(fNameHead),flush=True);
fContHead=BuildFileContentHead(pj1,pj2,cutRType,nLayer,nNeuron,\
     actiFun,opti,DW,k,c,n0,dn,nEpo,BS,rdSeed);
print('\n*****************************************\n',flush=True);
print('<File Content Head>:\n%s'%(fContHead),flush=True);
######## Load <OBS> points
nObs,PJ,Year,DD,xObs,yObs,zObs,bxObs,byObs,bzObs=\
     LoadObsData(pj1,pj2,cutRType,showinfo=True);
######## Load <Collocation> points
fileCol='input/Collocation_Random_minR1.00_maxR4.00_300000.txt';
nCol,xCol,yCol,zCol=LoadColData(fileCol,showinfo=True);
# %% # ! Model & Training
np.random.seed(rdSeed);
tf.random.set_seed(rdSeed);
layers = [3] + [nNeuron]*nLayer + [1];
obs_Data=[np.float32(xObs), np.float32(yObs),np.float32(zObs),\
     np.float32(bxObs), np.float32(byObs),np.float32(bzObs)]; 
col_Data=[np.float32(xCol), np.float32(yCol),np.float32(zCol)]; 
model=CurlCurl3D(layers,actiFun,opti,DW,obs_Data,col_Data);
model.train(nEpo,nBatPerEpo,BS,fNameHead,k,c,n0,dn);
# %% # ! Find best results with minimum loss in the last 10 iteration
true_loss=np.array(model.true_loss_log);
pkEpo=np.argmin(true_loss[nEpo-10:nEpo]) + nEpo-10+1;
print('Picked Epoch: %d'%(pkEpo),flush=True);
model.saver.restore(model.sess,save_path='./'+fNameHead+'/ckpt-%d'%(pkEpo));
# %% # ! Mismatch between Observation and PINN predicts
bx_obs_est, by_obs_est, bz_obs_est = model.predict_curl_A(xObs,yObs,zObs);
curlBx_col_est, curlBy_col_est, curlBz_col_est \
     = model.predict_curl_curl_A(xCol,yCol,zCol);
# Mean-Square-Error & Root-Mean-Square Error
print('\n ****** Training Reuslts ****** \n');
DispMSERMSE(bx_obs_est,bxObs,'bx','Gauss');
DispMSERMSE(by_obs_est,byObs,'by','Gauss');
DispMSERMSE(bz_obs_est,bzObs,'bz','Gauss');
DispMSERMSE(curlBx_col_est,0,'curlBx','Unnormaized Gauss/km | A/m^2');
DispMSERMSE(curlBy_col_est,0,'curlBy','Unnormaized Gauss/km | A/m^2');
DispMSERMSE(curlBz_col_est,0,'curlBz','Unnormaized Gauss/km | A/m^2');
# %% # ! Write Loss Logger to file
EPOCHs=np.array(range(1,nEpo+1))[:,None];
fileLossVars=fNameHead+'_LossVars.txt';
fid = open(fileLossVars, 'w');
wrtArray=np.hstack((EPOCHs,\
     np.array(model.loss_log)[:,None],\
     np.array(model.true_loss_log)[:,None],\
     np.array(model.bx_obs_loss_log)[:,None],\
     np.array(model.by_obs_loss_log)[:,None],\
     np.array(model.bz_obs_loss_log)[:,None],\
     np.array(model.curlBx_col_loss_log)[:,None],\
     np.array(model.curlBy_col_loss_log)[:,None],\
     np.array(model.curlBz_col_loss_log)[:,None],\
     np.array(model.w_obs_log)[:,None],\
     np.array(model.lr_log)[:,None]));
wrtFormat=['%07d','%6.3e','%6.3e',\
     '%6.3e','%6.3e','%6.3e',\
     '%6.3e','%6.3e','%6.3e',\
     '%6.3e','%6.3e'];
colHead='%5s %9s %9s '%('Epoch','Loss','TLoss')\
     +'%9s %9s %9s '%('LObs_bx','LObs_by','LObs_bz')\
     +'%9s %9s %9s '%('LCol_curlBx','LCol_curlBy','LCol_curlBz')\
     +'%9s %9s'%('wObs','LR');
wrtHead = fContHead + colHead;
np.savetxt(fid,wrtArray,fmt=wrtFormat,header=wrtHead);
fid.close();
print('\n*********************\n',flush=True);
print('Saving training info to file: %s'%(fileLossVars),flush=True);
