# Validation Guide

Run these checks before trusting the portable layer in a new project.

## 1. File And Source Check

From the bundle root:

```powershell
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -match "\\(build|__pycache__|reports|\.venv)\\" }
```

The command should not show checked-in generated artifacts in a clean copy.

## 2. Backend Self-Test

```powershell
cmake -S host\pc_backend -B host\pc_backend\build -G Ninja -DCMAKE_CXX_COMPILER=C:\Strawberry\c\bin\g++.exe
cmake --build host\pc_backend\build
host\pc_backend\build\vnema_backend.exe --self-test
```

Expected result: `self-test OK`.

## 3. Simulated Runtime And Observer

```powershell
host\pc_backend\build\vnema_backend.exe --simulate-can --mocap-sim --status-only --duration 2 --ids 0x101-0x118 --stream-state --log-dir reports > sim_out.jsonl
```

Acceptance:

- Backend exits cleanly.
- Report is written under `reports`.
- Expected replies equal received replies.
- Missed replies are zero.
- **Observer is alive:** most `robot_state` lines report `joint_current_valid:true`
  with non-zero, time-varying `joint_current_theta`. All-zero `q` across the run
  means the mocap→joint observer is broken (the legacy-bug signature). For example:

  ```powershell
  Select-String -Path sim_out.jsonl -Pattern '"joint_current_valid":true' | Measure-Object | Select-Object Count
  ```

  should report a large count (hundreds over a 2 s `--stream-state` run).

## 4. Python Smoke Checks

```powershell
python tools\can_probe.py --help
python tools\can_diag.py --help
python tools\can_dropout_tester.py --help
python tools\can_ota.py --help
python mocap\mocap.py --simulate --json --seconds 1
python examples\mpc_client_example.py --seconds 2
```

These do not require CAN hardware except for tests that open a COM port. The
example client runs the full command + read-state loop against the simulator; its
summary should report several hundred received states and a non-zero count of valid
joint estimates.

## 5. Hardware CAN Bring-Up

Keep outputs disabled.

```powershell
python tools\can_probe.py --ids 0x101-0x118 --sync --timeout 2 --port COM4 --tty-baudrate 2000000
python tools\can_diag.py --ids 0x101-0x118 --clear --port COM4 --timeout 5 --tty-baudrate 2000000
```

Acceptance:

- Expected boards respond.
- Diagnostics clear without persistent RX overflow or TX failure.

## 6. Runtime Broadcast Test

```powershell
python tools\can_dropout_tester.py --mode broadcast --order normal --ids 0x101-0x118 --prelude single --duration 30 --port COM4 --tty-baudrate 2000000 --log-dir reports
```

Acceptance for the validated 24-board path:

- Every selected board replies every cycle.
- Misses are zero or within the experiment's documented threshold.
- Compact status errors are zero.
- Post-test diagnostics show no persistent RX overflow.

## 7. Live Backend Status-Only

```powershell
host\pc_backend\build\vnema_backend.exe --mocap-sim --status-only --duration 10 --ids 0x101-0x118 --port COM4 --tty-baud 2000000 --log-dir reports
```

Acceptance:

- No unexpected replies.
- No duplicate replies.
- No compact status errors.
- Report and CSV are produced.

## 8. Mocap Validation

Simulated:

```powershell
python mocap\mocap.py --simulate --json --seconds 2 --rigid-ids 1000-1005
```

Live:

```powershell
python mocap\mocap.py --live --json --seconds 5 --server 192.168.1.100 --local 192.168.1.120 --rigid-ids 1000-1005 --multicast
```

Acceptance:

- Frame rate is near the configured Motive rate.
- Body IDs match the selected rigid bodies.
- Backend reports corrected mocap timestamp, raw timestamp, offset, frame rate, drop count, sample count, and update count during live runs.

## 9. Post-Port Criteria

A new PCB or host transport is not accepted until it passes:

- Backend self-test.
- Simulated runtime.
- Compact probe.
- Diagnostics clear/read.
- 150 Hz runtime broadcast or documented equivalent test.
- Sync-gated target step validation.
- Mocap timestamp correction validation when mocap is part of the experiment.
