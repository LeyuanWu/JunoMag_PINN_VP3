# JunoMag_PINN_VP3
Data and Code to reproduce the results in the manuscript: <br>
***Reconstruction of Jupiter's magnetic field using physics informed neural networks (PINN)***

> [!NOTE]  
> This is the `Vector Potential Version`
______________________________________________________________________________________________
## Input Data
- Juno vector magnetic data
  - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
  - Sampling rate: 30 sec;
  - Within 4.00 $R_j$, orbit 2 not used;
  - Orbit 1-33 (Prime mission);
  - Orbit 1-50 (Prime + Extended mission);
- `Spherical Harmonic Models` of Juno magnetic data
  - Format: **IND**, **SHC**, **g | h**, **n**, **m**;
  - `JRM33_N30`: degree 30 model using Juno prime mission orbit 1-33, orbit 2 not used (Connerney et al., 2022);
  - `Bloxham_N32`: degree 32 model using Juno prime mission orbit 1-33, orbit 2 not used (Bloxham et al., 2022);
- Collocation cloud:
  - Format: **x**, **y**, **z**
  - Random points within the region \[1.00 $R_j$, 4.00 $R_j$\];
  - Random points within the region \[0.80 $R_j$, 1.00 $R_j$\]; <br>
  *Generated based on the fact*: ***Normalized vector of Gaussian variables is uniformly distributed on the sphere***
______________________________________________________________________________________________
## Python Code and Jupyter Notebook
  - PINN training ***Above*** 1.00 $R_j$ (Obtain NN Jupiter Magnetic Model): [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PINN_VP3_JUNO.py)
  - PINN training ***Below*** 1.00 $R_j$ (Downward Continuation): [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PINN_VP3_DC_JUNO.py)
  - Table & Plot showing Juno Observation and Collocation Cloud distributions used in our PINN training [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_Obs_Col.ipynb)
  - Plot `PINN models` predicted $B_{\lambda}$, $B_{\theta}$, $B_{r}$ on multiple $R_j$:
    - ***Above*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_PINN_VP3_DC_JUNO.ipynb)
  - Plot Loss functions, Dynamic weights and Learning Rate
    - ***Above*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_LossFunc_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_LossFunc_PINN_VP3_DC_JUNO.ipynb)
  - Plot comparison of $B_{r}$ (or other components) predicted by `PINN models` and `SH models` on multiple $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_SHs_vs_PINNs.ipynb)
  - Plot comparison of Lowes Spectrums of `PINN models` and `SH models`: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_LowesSpectrum.ipynb)
  - Predict magnetic vector fields at ***Juno Observation Locations*** using `PINN models` and write to ascii file: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_JUNO.ipynb)
    - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
  - Predict magnetic vector fields at ***Multiple*** $R_j$ using `PINN models` and write to ascii file:
    - Format: **Lon**, **Lat**, $B_{\theta}$, $B_{\lambda}$, $B_{r}$, $|\mathbf{B}|$;
    - ***Above*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_DC_JUNO.ipynb)
  - Predict magnetic vector fields at ***User-Defined*** locations using `PINN models` and write to ascii file:
    - Format: **x**, **y**, **z**, **Bx**, **By**, **Bz**
    - ***Above*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_DC_JUNO.ipynb)
  - Predict magnetic vector fields at ***Juno Observation Locations*** & ***Multiple*** $R_j$ using `Spherical Harmonic Models` and write to ascii file: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_FromSHCs.ipynb)
  - Table showing the RMS errors of `Spherical Harmonic Models` computed at different subset of ***Juno Observation Orbits***: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_DataMissfit_SHs.ipynb)
  - Table & Plot showing the RMS errors of `PINN Models` `PINN33e` & `PINN50e` computed at different subset of ***Juno Observation Orbits***: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_DataMissfit_PINNs.ipynb)
  - Plot comparing RMS errors of `PINN Models`and `SH models` on each orbit of ***Juno Observation Orbits***: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_DataMissfit_SHs_vs_PINNs.ipynb)
  - Table & Plot showing `PINN models` predicted physical misfit <current density $|\mathbf{J}|$ > on multiple $R_j$
    - ***Above*** 1.00 $R_j$ (.vs. data missfit < $\delta |\mathbf{B}|$ >): [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_J_vs_dB_Rjs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_j$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_J_Rjs_PINN_VP3_DC_JUNO.ipynb)
______________________________________________________________________________________________
## Output NN Model
  - Heperparameters
    - Network shape: 6*40;
    - Activation Function: swish;
    - Optimizer: Adam;
    - Dynamic Weights: ON;
    - RAD parameter: k=1, c=0, n0=3000, dn=600;
    - Epochs: 12000;
    - Batch size: 10000;
  - Models (Last 10 models of each training)
    - ***Above*** 1.00 $R_j$: 
      - `PINN33e`: PJ01_33: model trained using ***Prime mission Juno data***;
      - `PINN50e`: PJ01_50: model trained using ***Prime + Extended mission Juno data***;
    - ***Below*** 1.00 $R_j$:
      - `PINN33i`: a downwards extrapolation model in $r<=1.00R_j$ which is fit to the 3-component magnetic field on $r=1.00R_j$ from PINNs model `PINN33e`;
      - `PINN50i`: a downwards extrapolation model in $r<=1.00R_j$ which is fit to the 3-component magnetic field on $r=1.00R_j$ from PINNs model `PINN50e`;
______________________________________________________________________________________________
## Output data
- `PINN models` training process data:
   - Data fitting and physics **Loss terms**, **Dynamic weights**, **Learning Rate**;
- `PINN models` and `Spherical Harmonic Models` predicted magnetic vector fields at ***Juno Observation Locations***:
   - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
- `PINN models` and `Spherical Harmonic Models` predicted *gridded* Jupiter magnetic vector fields at ***Multiple*** $R_j$:
   - Format: **Lon**, **Lat**, $B_{\theta}$, $B_{\lambda}$, $B_{r}$, $|\mathbf{B}|$;
   - Resolution: 1 $^{\circ}$ $\times$ 1 $^{\circ}$ with global coverage LON = $0.5:1:359.5$, LAT = $-89.5:1:89.5$;
   - 1.20 $R_j$, 1.15 $R_j$, 1.10 $R_j$, 1.05 $R_j$, 1.00 $R_j$, 0.95 $R_j$, 0.90 $R_j$, 0.85 $R_j$, 0.80 $R_j$;
______________________________________________________________________________________________
## References
- Connerney, J. E. P., Timmins, S., Oliversen, R. J., Espley, J. R., Joergensen, J. L., Kotsiaros, S., et al. (2022). A new model of Jupiter's magnetic field at the
completion of Juno's Prime Mission. Journal of Geophysical Research: Planets, 127, e2021JE007055. https://doi.org/10.1029/2021JE007055
- Bloxham, J., Moore, K. M., Kulowski, L., Cao, H., Yadav, R. K., Stevenson, D.J., et al. (2022). Differential rotation in Jupiter's interior revealed by simultaneous
inversion for the magnetic field and zonal flux velocity. Journal of Geophysical Research: Planets, 127, e2021JE007138. https://doi.org/10.1029/2021JE007138
