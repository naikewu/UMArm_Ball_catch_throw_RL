# Installation And Build Guide

Commands below assume Windows PowerShell and that the current directory is the copied `portable_sync_comm_layer` folder.

## Python Tools

Create or activate a Python environment, then install the required packages:

```powershell
python -m pip install -r requirements.txt
```

The Python tools need:

- `python-can` for SLCAN access.
- `pyserial` for COM port discovery.
- Tkinter from the Python standard library for GUIs.
- The OptiTrack NatNet Python client for live mocap. Place the SDK where `mocap/mocap.py` expects it or update `SDK_PYTHON_CLIENT`.

## Host Backend

Build the C++ backend with CMake/Ninja. The target compiles three translation
units: `main.cpp` (scheduler, CAN protocol, transports, mocap bridge, JSON IPC),
`umarm_fk.cpp` (forward kinematics), and `viz_shared_memory.cpp` (optional
visualization feed).

```powershell
cmake -S host\pc_backend -B host\pc_backend\build -G Ninja -DCMAKE_CXX_COMPILER=C:\Strawberry\c\bin\g++.exe
cmake --build host\pc_backend\build
host\pc_backend\build\vnema_backend.exe --self-test
```

Expected: `self-test OK`.

Run simulation without hardware (add `--stream-state` to emit one synchronized
`robot_state` per 150 Hz cycle instead of the 1 Hz heartbeat):

```powershell
host\pc_backend\build\vnema_backend.exe --simulate-can --mocap-sim --status-only --duration 2 --ids 0x101-0x118 --stream-state --log-dir reports
```

Run live SLCAN status-only with outputs disabled:

```powershell
host\pc_backend\build\vnema_backend.exe --mocap-sim --status-only --duration 10 --ids 0x101-0x118 --port COM4 --tty-baud 2000000 --log-dir reports
```

For live mocap from the backend (the backend launches `mocap/mocap.py` itself):

```powershell
host\pc_backend\build\vnema_backend.exe --mocap-live --mocap-python python --mocap-script mocap\mocap.py --mocap-server 192.168.1.100 --mocap-local 192.168.1.120 --mocap-rigid-ids 1000-1005 --ids 0x101-0x118 --port COM4 --stream-state --log-dir reports
```

## Example Controller Client

The bundle ships an MPC-style reference client that launches the backend, streams
synchronized `q`/`qdot`/pressure at 150 Hz, and publishes a target-pressure vector
each cycle. It uses only the Python standard library and runs against the simulator
with no hardware:

```powershell
python examples\mpc_client_example.py --seconds 3
```

Against real hardware and live mocap:

```powershell
python examples\mpc_client_example.py --live --port COM4 --mocap-server 192.168.1.100 --mocap-local 192.168.1.120 --enable-outputs --seconds 10
```

See [mpc_integration.md](mpc_integration.md) for the full JSON command/`robot_state`
contract this client is built on. There is no bundled operator GUI; the backend is
driven over its JSON stdin/stdout protocol (the example client is the reference
integration), with CLI diagnostics under `tools/`.

## OTA Tool

Launch the OTA GUI:

```powershell
python tools\can_ota.py
```

The portable default firmware image paths are:

- DT/big valve: `firmware/build_dt/Valve_not_embedded_XL.bin`.
- 7mm valve: `firmware/build_7mm/Valve_not_embedded_XL.bin`.

CLI dry run:

```powershell
python tools\can_ota.py --cli --ids 0x101 0x109 --dry-run
```

## ESP-IDF Firmware

The copied firmware source under `firmware/esp_idf` is the reference communication layer. In the original project, the validated builds were:

```powershell
$env:IDF_TOOLS_PATH="C:\ESP\ESP_tools"
$env:IDF_PYTHON_ENV_PATH="C:\ESP\ESP_tools\python_env\idf5.5_py3.11_env"
. "C:\ESP\ESP_container\v5.5.1\esp-idf\export.ps1"
idf.py -B build_dt -DVALVE_TYPE=1 build
idf.py -B build_7mm -DVALVE_TYPE=0 build
```

For a future project, copy or include `firmware/esp_idf` into the ESP-IDF application and implement the board-specific items in [hardware_porting.md](hardware_porting.md).

## SLCAN Notes

`bitrate` is the CAN bus bitrate. `tty_baudrate` or `--tty-baud` is the serial speed between PC and adapter. Keep the serial speed at 2 Mbps or faster for multi-board 150 Hz operation when the adapter supports it.
