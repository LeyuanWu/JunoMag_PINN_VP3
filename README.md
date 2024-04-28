# JunoMag_PINN_VP3
Data and Code to reproduce the results in the manuscript: <br>
***Reconstructions of Jupiter's magnetic field using physics-informed neural networks (PINN)***

> [!NOTE]  
> This is the `Vector Potential Version`
______________________________________________________________________________________________
## Input Data
- Juno vector magnetic data
  - Orbit 1-33 (Prime mission)
  - Orbit 1-50 (Prime + Extended mission)
  - Sampling rate: 30 sec
  - Within 4.00 $R_j$, orbit 2 not used
  - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
- `Spherical Harmonic Models` of Juno magnetic data
  - `JRM33_I30`: degree 30 model of (Connerney et al., 2022)
  - `Bloxham_I32`: degree 32 model of (Bloxham et al., 2022)
  - Format: **IND**, **SHC**, **g | h**, **n**, **m**
- Collocation cloud
  - Random points within the region $[1.00 R_J, 4.00 R_J]$
  - Random points within the region $[0.80 R_J, 1.00 R_J]$
  - Random points on the surface $r = 1.00 R_J$
  - Format: **x**, **y**, **z**
  *Generated based on the fact*: *Normalized vector of Gaussian variables is uniformly distributed on the sphere*
______________________________________________________________________________________________
## Python Code and Jupyter Notebook
  - PINN training ***Above*** 1.00 $R_J$ (Obtain NN Jupiter Magnetic Model): [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PINN_VP3_JUNO.py)
  - PINN training ***Below*** 1.00 $R_J$ (Downward Continuation): [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PINN_VP3_DC_JUNO.py)
  - Table & Plot showing Juno Observation and Collocation Cloud distributions used in our PINN training [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_Obs_Col.ipynb)
  - Plot `PINN models` predicted $B_{\lambda}$, $B_{\theta}$, $B_{r}$ on multiple $R_J$:
    - ***Above*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_PINN_VP3_DC_JUNO.ipynb)
  - Plot Loss functions, Dynamic weights and Learning Rate
    - ***Above*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_LossFunc_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_LossFunc_PINN_VP3_DC_JUNO.ipynb)
  - Plot comparison of $B_{r}$ (or other components) predicted by `PINN models` and `SH models` on multiple $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_SHs_vs_PINNs.ipynb)
  - Plot comparison of Lowes Spectrums of `PINN models` and `SH models`: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_LowesSpectrum.ipynb)
  - Predict magnetic vector fields at ***Juno Observation Locations*** using `PINN models` and write to ascii file: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_JUNO.ipynb)
    - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
  - Predict magnetic vector fields at ***Multiple*** $R_J$ using `PINN models` and write to ascii file:
    - Format: **Lon**, **Lat**, $B_{\theta}$, $B_{\lambda}$, $B_{r}$, $|\mathbf{B}|$;
    - ***Above*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_DC_JUNO.ipynb)
  - Predict magnetic vector fields at ***User-Defined*** locations using `PINN models` and write to ascii file:
    - Format: **x**, **y**, **z**, **Bx**, **By**, **Bz**
    - ***Above*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_PINN_VP3_DC_JUNO.ipynb)
  - Predict magnetic vector fields at ***Juno Observation Locations*** & ***Multiple*** $R_J$ using `Spherical Harmonic Models` and write to ascii file: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PredictBs_FromSHCs.ipynb)
  - Table showing the RMS errors of `Spherical Harmonic Models` computed at different subset of ***Juno Observation Orbits***: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_DataMissfit_SHs.ipynb)
  - Table & Plot showing the RMS errors of `PINN Models` `PINN33e` & `PINN50e` computed at different subset of ***Juno Observation Orbits***: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_DataMissfit_PINNs.ipynb)
  - Plot comparing RMS errors of `PINN Models`and `SH models` on each orbit of ***Juno Observation Orbits***: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_DataMissfit_SHs_vs_PINNs.ipynb)
  - Table & Plot showing `PINN models` predicted physical misfit <current density $|\mathbf{J}|$ > on multiple $R_J$
    - ***Above*** 1.00 $R_J$ (.vs. data missfit < $\delta |\mathbf{B}|$ >): [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_J_vs_dB_Rjs_PINN_VP3_JUNO.ipynb)
    - ***Below*** 1.00 $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Table_Plot_J_Rjs_PINN_VP3_DC_JUNO.ipynb)
______________________________________________________________________________________________
## Output NN Models
  - Models (Last 10 models of each training)
    - ***Above*** 1.00 $R_J$: 
      - `PINN33e`: model trained using Orbit 1-33
      - `PINN50e`: model trained using Orbit 1-50
    - ***Below*** 1.00 $R_J$:
      - `PINN33i`: downwards extrapolation model of `PINN33e` in $[0.80 R_J, 1.00 R_J]$
      - `PINN50i`: downwards extrapolation model of `PINN50e` in $[0.80 R_J, 1.00 R_J]$
______________________________________________________________________________________________
## Output data
- `PINN models` and `Spherical Harmonic Models` predicted magnetic vector fields at *Juno Observation Locations*:
   - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
- `PINN models` and `Spherical Harmonic Models` predicted *gridded* Jupiter magnetic vector fields at Multiple $R_J$:
   - Format: **Lon**, **Lat**, $B_{\theta}$, $B_{\lambda}$, $B_{r}$, $|\mathbf{B}|$;
- `PINN models` predicted *gridded* data of electric currents ($\mathbf{J}$) at Multiple $R_J$:
   - Format: **Lon**, **Lat**, $J_{\theta}$, $J_{\lambda}$, $J_{r}$, $|\mathbf{J}|$;
- `PINN models` training process data:
   - Data and physics *Loss terms*, *Dynamic weights*, *Learning Rate*;
______________________________________________________________________________________________
## References
- Connerney, J. E. P., Timmins, S., Oliversen, R. J., Espley, J. R., Joergensen, J. L., Kotsiaros, S., et al. (2022). A new model of Jupiter's magnetic field at the completion of Juno's Prime Mission. Journal of Geophysical Research: Planets, 127, e2021JE007055. https://doi.org/10.1029/2021JE007055
- Bloxham, J., Moore, K. M., Kulowski, L., Cao, H., Yadav, R. K., Stevenson, D.J., et al. (2022). Differential rotation in Jupiter's interior revealed by simultaneous inversion for the magnetic field and zonal flux velocity. Journal of Geophysical Research: Planets, 127, e2021JE007138. https://doi.org/10.1029/2021JE007138
- Edwards, T. M., Bunce, E. J., & Cowley, S. W. H. (2001). A note on the vector potential of Connerney et al.'s model of the equatorial current sheet in Jupiter's magnetosphere. Planetary and Space Science, 49, 1115-1123. https://doi.org/10.1016/S0032-0633(00)00164-1
