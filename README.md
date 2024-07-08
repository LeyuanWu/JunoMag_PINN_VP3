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
  - Within 4.00 $R_J$, orbit 2 not used
  - Format: **PJ**, **Year**, **Decimal-Day**, **x**, **y**, **z**, **Bx**, **By**, **Bz**
- `Spherical Harmonic Models` of Juno magnetic data
  - `JRM33_I30`: degree 30 model of (Connerney et al., 2022)
  - `Bloxham_I32`: degree 32 model of (Bloxham et al., 2022)
  - Format: **IND**, **SHC**, **g | h**, **n**, **m**
- Collocation cloud
  - Random points within the region $[1.00 R_J, 4.00 R_J]$
  - Random points within the region $[0.80 R_J, 1.00 R_J]$
  - Random points on the surface $r = 1.00 R_J$
  - Format: **x**, **y**, **z**  <br>
  *Generated based on the fact*: *Normalized vector of Gaussian variables is uniformly distributed on the sphere*
______________________________________________________________________________________________
## Python Code and Jupyter Notebook
  - Training diagnostics from training model `PINN50e`: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_LossFunc_PINN_VP3.ipynb)
  - Plot comparison of RMS errors of `PINN Models` and `SH models` on each orbit of *Juno Observation Orbits*: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/DataMisfit_SHs_vs_PINNs_PJ01_50_binByPJ.ipynb)
  - Plot showing `PINN models` predicted physical misfit (current density $|\mathbf{J}|$) on Multiple $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_JnedRjs.ipynb)
  - Plot comparison of $B_{r}$ predicted by `PINN models` and `SH models` on Multiple $R_J$: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Plot_BnedRjs_SHs_vs_PINNs.ipynb)
  - Plot comparison of Lowes spectrums of `PINN models` and `SH models`: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/SHs_Spectrum2D_Lowes.ipynb)
  - Table showing the RMS errors of `Spherical Harmonic Models` with increasing degree $n$ at different subset of *Juno Observation Orbits*: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/DataMisfit_SHs_MagDisc.ipynb)
  - Table showing the RMS errors of `PINN Models` at different subset of *Juno Observation Orbits*: [Jupyter Notebook](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/DataMisfit_PINNs.ipynb)
  - PINN training ***Above*** 1.00 $R_J$ (NN Jupiter Magnetic Model): [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PINN_VP3_JUNO.py)
  - PINN training ***Below*** 1.00 $R_J$ (Downward Continuation): [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/PINN_VP3_DC_JUNO.py)
  - Predict magnetic vector fields at *Juno Observation Locations* using `PINN models` and write to ascii file: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BxyzOBS_fromPINN_VP3.py)
  - Predict magnetic vector fields at Multiple $R_J$ using `PINN models` and write to ascii file:
    - ***Above*** 1.00 $R_J$: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BnedRjs_fromPINN_VP3.py)
    - ***Below*** 1.00 $R_J$: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BnedRjs_fromPINN_VP3_DC.py)
  - Predict electric currents at Multiple $R_J$ using `PINN models` and write to ascii file:
    - ***Above*** 1.00 $R_J$: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_JnedRjs_fromPINN_VP3.py)
    - ***Below*** 1.00 $R_J$: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_JnedRjs_fromPINN_VP3_DC.py)
  - Predict magnetic vector fields at *Interface* $r=1.00 R_J$ using `PINN models` and write to ascii file:
    - ***Above*** 1.00 $R_J$: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BxyzITFC_fromPINN_VP3.py)
    - ***Below*** 1.00 $R_J$: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BxyzITFC_fromPINN_VP3_DC.py)
  - Predict magnetic vector fields at *Juno Observation Locations* using `Spherical Harmonic Models` and write to ascii file: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BxyzOBS_fromSH_MagDisc.py)
  - Predict magnetic vector fields at Multiple $R_J$ using `Spherical Harmonic Models` and write to ascii file: [Python Code](https://github.com/LeyuanWu/JunoMag_PINN_VP3/blob/main/Predict_BnedRjs_fromSH.py)
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
