##################################################################
# # ! Juno Magnetic Modelling using <Vector Potential> with <3 NNs>
# Table & Plot showing the RMS errors of `PINN Models` 
# computed at different subset of ***Juno Observation Orbits***
##################################################################
# %%
# # ! Setup
import numpy as np;
from LW_DataReadWrite import *;
# def prRed(skk): print("\033[91m {}\033[00m" .format(skk));
# def prGreen(skk): print("\033[92m {}\033[00m" .format(skk));
# def prCyan(skk): print("\033[96m {}\033[00m" .format(skk));
# %%
# # ! PINN Model Estimations of [Bx,By,Bz] at <Multiple Different Obs Dataset>
# TODO ******************** Parameters ******************** #
cstRj=71492;        # Jupiter radius in km
GS2nT=1e5;          # Gauss to nt
# TODO ********************** end ************************* #
######## Test Juno OBS dataset
FILEOBSs = ['input/Connerney_PJ01_33_4.0Rj.txt',      # <Connerney Obs dataset PJ01-33>
            'input/Juno_PJ01_33_4.0Rj.txt',           # <Our Obs dataset PJ01-33>
            'input/Juno_PJ01_50_4.0Rj.txt'];          # <Our Obs dataset PJ01-50>  
OBSNAMEs = ['CnyObs33',
            'OurObs33',
            'OurObs50'];
######## PINNs models
PINNFILEs = [
             [
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_050_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_050_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN06_050_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_050_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_050_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN06_050_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN07_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN07_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN07_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN07_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN07_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN07_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN08_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN08_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_33_4.0Rj_NN08_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ],
             [
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN08_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_ConnerneyOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN08_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS33.txt',
              'output/JUNO_PINN_VP3_PJ01_50_4.0Rj_NN08_040_swish_Adam_DW1_RADk1c0n3000d600_nEpo0012000_BS0010000_Seed12345_EstBxyz_OurOBS50.txt'
             ]
            ];
PINNNAMEs = ['NN06_040_PINN33e',
             'NN06_040_PINN50e',
             'NN06_050_PINN33e',
             'NN06_050_PINN50e',
             'NN07_040_PINN33e',
             'NN07_040_PINN50e',
             'NN08_040_PINN33e',
             'NN08_040_PINN50e'];
# TODO ********************** end ************************* #
for iPINN in range(len(PINNFILEs)):
     pinnName = PINNNAMEs[iPINN];
     print('\n*********************\n',flush=True);
     print('RMS of \033[41m %s \033[0m evaluated at multiple different Juno <OBS> dataset:'%(pinnName),flush=True);
     for iFILEOBS in range(len(FILEOBSs)):
          fileEst = PINNFILEs[iPINN][iFILEOBS];
          dataEst=np.loadtxt(fileEst,skiprows=1);
          estBx=dataEst[:,6:7]; estBy=dataEst[:,7:8]; estBz=dataEst[:,8:9];
          estBNorm=np.sqrt(estBx**2+estBy**2+estBz**2);
          fileObs = FILEOBSs[iFILEOBS];
          nObs,PJ,Year,DD,xObs,yObs,zObs,bxObs,byObs,bzObs = LoadObsFile(fileObs,showinfo=False);
          refBNorm = np.sqrt(bxObs**2+byObs**2+bzObs**2);
          print('%9s, '%(OBSNAMEs[iFILEOBS]),end='',flush=True);
          rms = np.sqrt(np.mean((estBNorm-refBNorm)**2));
          print('rms = %10.1f nT; '%(GS2nT*rms),flush=True);
     print('',flush=True); # <br>