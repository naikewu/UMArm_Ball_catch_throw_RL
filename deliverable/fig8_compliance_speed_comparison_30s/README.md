# Figure-eight tip speed comparison

[Tip-only four-panel MP4](fig8_tip_speed_comparison_30s_seed20260912.mp4)
[Tip-and-compliance four-panel MP4](fig8_tip_and_compliance_speed_comparison_30s_seed20260912.mp4)

All panels retain the same fitted twin, Koopman MPPI settings, 35 mm by 18 mm tip path, seed, and 18-second compliance schedule. The path traversal rate alone changes.

| Speed | Tip RMS | Tip p95 | Tip max | True xy compliance median error |
| ---: | ---: | ---: | ---: | ---: |
| 0.5x | 0.984 mm | 1.831 mm | 3.027 mm | 2.882% |
| 1x | 1.527 mm | 2.902 mm | 4.397 mm | 3.143% |
| 1.5x | 2.106 mm | 3.767 mm | 5.042 mm | 2.952% |
| 2x | 2.699 mm | 5.124 mm | 6.228 mm | 3.232% |

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m control.render_fig8_speed_comparison --duration-s 30 --fps 10 --compliance-head data\compliance_tangent_wide_5000\head.npz --with-compliance --reuse-traces
```

The video shows nominal digital-twin tracking only. It is not a real-hardware speed qualification.
