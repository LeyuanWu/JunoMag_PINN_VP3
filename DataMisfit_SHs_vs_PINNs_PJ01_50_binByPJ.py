##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Plot showing the RMS errors of 'SH models' & `PINN Models` 
# computed at each perijove orbit of ***Juno Observation Orbits***
##################################################################
# %%
# # ! Setup
import numpy as np;
import matplotlib.pyplot as plt
from LW_DataReadWrite import *;
# %%
# # ! RMS of PINN Models Estimations of |B| at <Obs>: at individual orbit
# TODO ******************** Parameters ******************** #
cstRj=71492;      # Jupiter radius in km
GS2nT=1e5;        # Gauss to nt
# TODO ********************** end ************************* #
fileObs='input/Juno_PJ01_50_4.0Rj.txt';
nObs,PJ,Year,DD,xObs,yObs,zObs,bxObs,byObs,bzObs = LoadObsFile(fileObs,showinfo=False);
RObs=np.sqrt(xObs**2+yObs**2+zObs**2); 
RinRj=RObs/cstRj;
bNormObs = np.sqrt(bxObs**2 + byObs**2 + bzObs**2);
FILEs = ['output/JRM33_I30MDa_nmax18_EstBxyz_OurObs50.txt',\
         'output/JRM33_I30MDa_nmax30_EstBxyz_OurObs50.txt',\
         'output/Bloxham_I32MDb_nmax18_EstBxyz_OurObs50.txt',\
         'output/Bloxham_I32MDb_nmax32_EstBxyz_OurObs50.txt',\
         'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurObs50.txt',\
         'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurObs50.txt',];
MODELs = ['JRM33 (n=18)','JRM33 (n=30)','Baseline (n=18)','Baseline (n=32)','PINN33e','PINN50e'];

nMODEL = len(MODELs); 
pjs = np.unique(PJ); npj = pjs.size;
RMS = [[] for i in range(nMODEL)]; # RMS on each orbit
E2 = [[] for i in range(nMODEL)];  # E2 error (L2 norm relative error %) on each orbit
ERRORs = [[] for i in range(nMODEL)];
for iMODEL in range(nMODEL):
     modelName = MODELs[iMODEL];
     fileEst = FILEs[iMODEL];
     dataEst=np.loadtxt(fileEst,skiprows=1);
     estBx=dataEst[:,6:7]; estBy=dataEst[:,7:8]; estBz=dataEst[:,8:9];
     estBNorm=np.sqrt(estBx**2+estBy**2+estBz**2);
     print('\n*********************\n',flush=True);
     print('RMS of %s evaluated at Juno <OBS>:'%(modelName),flush=True);
     for ipj in range(npj):
          curpj = pjs[ipj];
          pkInd = np.logical_and((PJ==curpj),(RinRj<=4.0));
          refBx_pj = bxObs[pkInd]; refBy_pj = byObs[pkInd]; refBz_pj = bzObs[pkInd];
          refBNorm_pj = bNormObs[pkInd];
          estBx_pj = estBx[pkInd]; estBy_pj = estBy[pkInd]; estBz_pj = estBz[pkInd];
          estBNorm_pj = estBNorm[pkInd];
          # Delta |B|
          errBNorm1 = estBNorm_pj - refBNorm_pj;
          # |Delta B|
          errBNorm2 = np.sqrt((estBx_pj-refBx_pj)**2+(estBy_pj-refBy_pj)**2+(estBz_pj-refBz_pj)**2);
          # Delta Bx, By, Bz
          err_dBxyz = np.concatenate((estBx_pj-refBx_pj,estBy_pj-refBy_pj,estBz_pj-refBz_pj),axis=0);
          print('PJ[%02d], NOB: %4d, '%(curpj,np.count_nonzero(pkInd)),end='',flush=True);
          rms = np.sqrt(np.mean(err_dBxyz**2));
          e2 = rms/np.sqrt(np.mean(refBNorm_pj**2));
          print('rms = %8.1f nT; E2 = %8.4f %%'%(GS2nT*rms,100*e2),flush=True);
          RMS[iMODEL].append(GS2nT*rms);
          E2[iMODEL].append(100*e2);
          ERRORs[iMODEL].append(GS2nT*err_dBxyz);
     print('',flush=True); # <br>
# %%
# # ! Plot of PINN Models Estimations Error of |B| at <Obs>: at individual orbit
pkMODELS = np.array([1,4,5,6])-1;
# pkMODELS = np.array([1,2,3,4,5,6])-1;
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), dpi=150, layout='constrained');
yLables=['RMS error of $|\mathbf{B}|$ (nT)','$E_2$ of $|\mathbf{B}|$ (%)'];
COLORs =['#00FFFF','C1','C2','#FF00FF','#FF0000','#0000FF'];
LSs = ['-','-','-','-','-','-'];
MARKERs =['o','o','o','o','o','o'];
for iMODEL in pkMODELS:
     modelName = MODELs[iMODEL];
     axs[0].step(pjs,RMS[iMODEL],color=COLORs[iMODEL],linestyle=LSs[iMODEL],where='mid',label=modelName);
     axs[0].plot(pjs,RMS[iMODEL], MARKERs[iMODEL], color=COLORs[iMODEL],linestyle='',alpha=0.5);
     axs[0].set_ylim(bottom=0,top=3200);
     axs[1].step(pjs,E2[iMODEL], color=COLORs[iMODEL],linestyle=LSs[iMODEL],where='mid',label=modelName);
     axs[1].plot(pjs,E2[iMODEL],  MARKERs[iMODEL], color=COLORs[iMODEL],linestyle='',alpha=0.5);
     axs[1].set_ylim(bottom=0,top=1.2);
for iax in range(len(axs)):
     axs[iax].axvline(33.5,color='black',linestyle='--');
     axs[iax].set_xticks(pjs);
     axs[iax].xaxis.grid(True);
     axs[iax].yaxis.grid(True);
     axs[iax].set_xlabel('PJ');
     axs[iax].set_ylabel(yLables[iax]);
     axs[iax].legend(loc='best', ncol = pkMODELS.size//2)  # Add a legend
     axs[iax].tick_params(axis='x',which='major',labelrotation=45,labelsize='small');
plt.savefig('NN06_040_Plot_DataMisfit_SHs_vs_PINNs_OurObs50_binByPJ.png',dpi=300);
plt.show();