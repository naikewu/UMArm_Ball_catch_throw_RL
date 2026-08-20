# natnet_sdk — vendored third-party code

These three files are the NaturalPoint **NatNet SDK** Python client, copied here
**verbatim except for local patch 1 below** (`MoCapData.py` and
`DataDescriptions.py` remain byte-identical to the upstream copy). They are not
ours; do not reformat, re-lint, or "clean up" anything in them. Local fixes, if
ever unavoidable, go in a patch documented in this file so the next SDK drop can
be re-applied mechanically.

| File | Bytes | SHA-256 |
|------|------:|---------|
| `NatNetClient.py` (patch 1 applied) | 81195 | `abdc452d9b26aa8cdc3fbb2e873c5f1bef2a3ae448d6b8381440d421484c9f5c` |
| `MoCapData.py`        | 39812 | `c1dcbd8324d55f671fdf85fcb3a77207a50e3aa1a90014b014f9549dda7157b3` |
| `DataDescriptions.py` | 32540 | `d3dab43d046893b1eae6c4dfb54e77a8045bfa0ca18d3cad7ed9861105f83ebc` |

`NatNetClient.py` as shipped upstream (before patch 1) was 80727 bytes,
SHA-256 `9a52cb7b8e455b8fc8bffb75468b29be628de9e166e1bf19bad0ec1fcbd84e51` —
keep this line so the next SDK drop can be diffed against the true upstream.

Verify at any time (from this folder):

```bash
sha256sum NatNetClient.py MoCapData.py DataDescriptions.py
```

## Where they came from

Copied on **2026-08-11** from the lab's legacy research repo, which had itself
vendored them from the SDK install:

```
UMArm_compliance_TRO/mocap_to_config/natnet_sdk/{NatNetClient,MoCapData,DataDescriptions}.py
```

(full path on the bench PC:
`C:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMArm_compliance_TRO\mocap_to_config\natnet_sdk\`)

The hashes above match both that folder **and** the identical copies at the
legacy repo root (`UMArm_compliance_TRO/{NatNetClient,MoCapData,DataDescriptions}.py`),
which is what `kinematics_mp`/`mocap_natnet_receiver_routine.py` imported. So the
bytes here are the ones every mocap result in the lab's TRO data was produced
with — that is the reason for vendoring rather than pip-installing something
newer.

Upstream origin: the `NatNetSDK/Samples/PythonClient` directory of a NaturalPoint
NatNet SDK release (Motive's own Python sample client), Apache-2.0, "Copyright ©
2018 Naturalpoint" per the header of each file. The files carry no version
string; the depacketizer handles asset/skeleton data added in the NatNet 4.x
bitstream, so treat them as a 4.x-era drop and re-check against Motive's own
sample client before assuming support for a newer bitstream.

## Why they are here at all

`mocap_rx.MocapRx` needs a NatNet client to receive Motive's UDP stream. This is
vendoring a third-party SDK into our tree — *not* importing the legacy research
repo at runtime, which the design forbids. The re-implemented parts
(`mocap_to_q.py`, `mocap_constants.py`, `mocap_rx.py`) are ours and live one
directory up.

## Local patches

### Patch 1 — `mocap_data_listener` hook (2026-08-11)

The SDK's `__unpack_mocap_data` parses everything the marker-frame work needs
(`MarkerSetData`, `LabeledMarkerData`, the suffix's `tracked_models_changed`
bit) and then throws the `MoCapData` object away — the stock listeners only
hand out per-rigid-body poses and a summary dict. Patch 1 adds one optional
listener that receives the fully parsed frame, per `docs/marker_frame_design.md`
§3 (decision D1). Behaviour is unchanged when the listener is unset (the
default); when set, it fires on the SDK's data thread in rigid-bodies →
mocap_data → new_frame order, before and independent of the
`new_frame_listener` guard. `mocap_rx.MocapRx` hooks it in `start()`.

The exact diff (against the 80727-byte upstream file; both hunks preserve the
file's CRLF line endings and UTF-8 BOM):

```diff
--- a/UMArm_MOCAP/natnet_sdk/NatNetClient.py
+++ b/UMArm_MOCAP/natnet_sdk/NatNetClient.py
@@ -83,6 +83,10 @@ class NatNetClient:
         self.rigid_body_listener = None
         self.new_frame_listener  = None
 
+        # Set this to a callback method of your choice to receive the fully parsed
+        # MoCapData object once per frame. [local patch 1 -- see PROVENANCE.md]
+        self.mocap_data_listener = None
+
         # Set Application Name
         self.__application_name = "Not Set"
 
@@ -908,6 +912,10 @@ class NatNetClient:
         offset += rel_offset
         mocap_data.set_suffix_data(frame_suffix_data)
 
+        # Send the fully parsed frame to any listener, before and independent of
+        # the new_frame_listener below. [local patch 1 -- see PROVENANCE.md]
+        if self.mocap_data_listener is not None:
+            self.mocap_data_listener(mocap_data)
 
         timecode = frame_suffix_data.timecode
         timecode_sub= frame_suffix_data.timecode_sub
```

To re-apply on a future SDK drop: verify the new drop's `NatNetClient.py`
against upstream, apply the two hunks (the anchors are the listener block in
`__init__` and the `set_suffix_data` call at the end of `__unpack_mocap_data`),
then update the byte count and SHA-256 in the table above.

## Gotchas that follow from copying them verbatim

* **Flat imports.** `NatNetClient.py` does `import MoCapData` /
  `import DataDescriptions`, so this directory must be on `sys.path` before
  importing it. `MocapRx._import_natnet()` does exactly that, lazily, at
  `start()` — see the docstring there for why it is not done at module import.
* **Scalar-last quaternions.** The rigid-body listener hands out
  `[x, y, z, w]`. `mocap_to_q.quat_xyzw_to_matrix` matches that order.
* **A UTF-8 BOM** heads `NatNetClient.py`. It is part of the original bytes and
  is kept, which is why the file hashes as it does.
* **The SDK owns its own threads.** `run()` starts them; `shutdown()` stops
  them. Its listeners fire on those threads, which is what dictates the locking
  in `mocap_rx.py`.
