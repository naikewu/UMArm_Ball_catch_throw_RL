# Large horizontal-to-vertical compliance changes

The target Cxx and Cyy alternate between 0.054 and 0.098 m/N, giving a maximum
target aspect ratio of 1.815:1 (previous video: 1.30:1). They vary in opposite
phases: horizontal ellipse, near circle, vertical ellipse, then back again.
Both target and true ellipses use the same isotropic display scale (0.2 N)
and are centred on the actual tip. No temporal smoothing is used.

The tip path and its timing are unchanged from the previous revised video.
Only the compliance phase is slowed by a factor of two (nominal period 18 s
instead of 9 s), to accommodate the existing 5 psi/s common-pressure slew
limit. The first wide-range trial at 9 s is preserved in
`../fig8_compliance_wide_30s`; it had 8.65% median true xy compliance error.

Control uses `data/compliance_tangent_wide_5000/head.npz`, a polynomial joint
stiffness fit from 5000 current-twin samples, with a 4000/1000 train/holdout
split and 0.3--14.5 psi pair means. Negative line pressures are excluded from
the dataset. Holdout compliance error: median 0.0701%, p90 0.1858%.

Blue ellipses are calculated separately using MuJoCo torque derivatives at
the actual simulated joint state and pressure. The definition is local
frozen-pressure static compliance at the final plate centre, not the dynamic
admittance of a hardware feedback loop. This remains an offline experiment.

## Results

- Actual maximum horizontal and vertical aspect ratios: both approximately 1.681.
- Horizontal example: 6.7 s. Vertical example: 15.4 s.
- True xy compliance relative error: median 3.143%, p90 4.326%.
- Cxx/Cyy target-versus-true correlation: 0.9903 / 0.9893.
- Drawing tip RMS: 1.527 mm; p95: 2.902 mm; maximum: 4.397 mm.
- Maximum requested antagonist pair sum: 29.224 psi, below the 30 psi cap.
- The larger target does not track as accurately as the older, smaller-range
  target (0.925% median compliance error, 1.071 mm tip RMS). The increased
  anisotropy and this accuracy tradeoff are retained in the raw data.
- 29 focused regression tests passed, including independent compliance-period
  control without changing the tip reference trajectory.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m digital_twin.train_tangent_compliance --samples 5000 --mean-min-psi 0.3 --mean-max-psi 14.5 --out data/compliance_tangent_wide_5000
.\.venv\Scripts\python.exe -m control.render_fig8_compliance_gif --duration-s 30 --fps 10 --compliance-soft 0.098 --compliance-hard 0.054 --compliance-period-s 18 --compliance-head data/compliance_tangent_wide_5000/head.npz --out-dir deliverable/fig8_compliance_wide_slow_30s --mp4
```

The 30 s MP4 is the main deliverable; the GIF, PNG, trajectory NPZ,
independent-compliance NPZ and JSON metrics are retained alongside it.
