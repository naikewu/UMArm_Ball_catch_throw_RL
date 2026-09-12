# Larger horizontal/vertical compliance variation

The new target changes Cxx and Cyy in opposite phases between 0.054 and
0.098 m/N, increasing the target peak aspect ratio from 1.30 to 1.815.
The xy tip trajectory, timing, controller weights, and display force remain
the same as the previous revised experiment. Both ellipses are centred on
the actual tip and use the same isotropic 0.2 N display scale. No anisotropic
display stretching or temporal smoothing is applied.

To cover the lower common pressures needed for stronger anisotropy, a new
polynomial stiffness head was fitted with 5000 samples (4000 training, 1000
holdout). The common-pressure range is 0.3--14.5 psi. Individual pressures
are nonnegative and antagonist sums remain below 30 psi. The planner takes
its mean-pressure bounds from the selected head's training metadata.
Holdout compliance relative error: median 0.0701%, p90 0.1858%, max 0.8875%.

The controller uses this fitted head for compliance prediction. Blue ellipses
are independently calculated from the MuJoCo mechanics at the recorded
actual q and pressures. The definition remains local, frozen-pressure static
compliance at the final plate centre, not dynamic closed-loop admittance.
This is an offline simulation; 150 Hz wall-clock control is not established.

```powershell
.\.venv\Scripts\python.exe -m digital_twin.train_tangent_compliance --samples 5000 --mean-min-psi 0.3 --mean-max-psi 14.5 --out data/compliance_tangent_wide_5000
.\.venv\Scripts\python.exe -m control.render_fig8_compliance_gif --duration-s 30 --fps 10 --compliance-soft 0.098 --compliance-hard 0.054 --compliance-head data/compliance_tangent_wide_5000/head.npz --out-dir deliverable/fig8_compliance_wide_30s --mp4
```

The MP4, GIF, PNG, trajectory NPZ, independent-compliance NPZ, and JSON metrics
are saved here. The previous video and its model remain unchanged.
