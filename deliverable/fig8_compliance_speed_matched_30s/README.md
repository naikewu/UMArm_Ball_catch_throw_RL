# Figure-eight tip speed comparison

[Tip-and-compliance four-panel MP4](fig8_tip_and_compliance_speed_comparison_30s_seed20260912.mp4)

All panels retain the same fitted twin, Koopman MPPI settings, 35 mm by 18 mm tip path, seed, and compliance target range. The compliance reference uses the same phase and speed scale as the tip path.

| Speed | Tip RMS | Tip p95 | Tip max | True xy compliance median error |
| ---: | ---: | ---: | ---: | ---: |
| 0.5x | 0.984 mm | 1.831 mm | 3.027 mm | 2.882% |
| 1x | 1.360 mm | 2.629 mm | 4.218 mm | 8.654% |
| 1.5x | 1.754 mm | 3.007 mm | 3.830 mm | 14.162% |
| 2x | 2.423 mm | 4.632 mm | 5.666 mm | 16.624% |

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m control.render_fig8_speed_comparison --duration-s 30 --fps 10 --compliance-head data\compliance_tangent_wide_5000\head.npz --with-compliance --reuse-traces
```

The video shows nominal digital-twin tracking only. It is not a real-hardware speed qualification.
