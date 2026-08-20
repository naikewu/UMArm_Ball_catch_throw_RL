# CAN-arm live mocap verification — 2026-08-20 (BLOCKED)

Session 17:35–17:52 local, `WS\.venv\Scripts\python.exe` (CPython 3.13.13). Passive
receive only: COM58 untouched, the Kinova never contacted; the only bytes transmitted
were NatNet NAT_CONNECT / NAT_PING packets to 192.168.1.100:1510.

**Verdict: FAILED — blocked, not broken.** No NatNet stream reached this PC. The
roster is empty, no CAN-arm `q` was observed, no chain was measured, and
`canarm_params.MEASURED` stays `False`. The fault was localised to the Motive host
rather than to anything in this workspace.

## 1. ID census — the roster is empty

`hw_tests/mocap_census.py` drives the vendored SDK directly and accepts every id
(`MocapRx` exists to *drop* out-of-block ids — correct for control, useless for a
census).

| window | frames | rigid bodies | labeled markers | marker sets |
|---|---|---|---|---|
| 10 s | 0 | none | none | none |
| 12 s | 0 | none | none | none |
| 30 s (raw socket) | 0 datagrams | — | — | — |

`NatNetClient.run()` returned true every time — exactly the silent failure the
receiver's docstring warns about: it reports the sockets bound and the group joined,
and says nothing about whether a server is talking.

### Where the fault is, in four passes

1. **The SDK is not the obstacle.** A raw `SOCK_DGRAM` on 239.255.42.99:1511 with the
   group joined on 192.168.1.120, and separately a unicast bind on
   192.168.1.120:1511, both saw 0 datagrams in 4 s.
2. **Motive's streaming engine is not serving.** NAT_PING to 192.168.1.100:1510 drew
   no reply, repeated every 2 s for 30 s — a NatNet server answers a ping whatever
   its streaming settings. ICMP to that host succeeds; its ARP entry reads
   `Reachable` (MAC `4C-D7-17-A8-08-0F`), so the host itself is up.
3. **The join did not attach to the wrong interface.** The group was joined on every
   local IPv4 address at once (192.168.1.120, 35.3.91.18, 127.0.0.1) across ports
   1511/1512/1513 simultaneously. Still 0 in 12 s.
4. **No second Motive host.** `Get-NetNeighbor` lists exactly two unicast neighbours
   on the subnet: 192.168.1.100 and the Kinova. A NAT_PING broadcast to
   192.168.1.255 also went unanswered.

Local explanations closed off: the Windows firewall carries four enabled inbound
Allow rules for `python.exe`, and `Get-NetUDPEndpoint` shows no process bound to
1510–1513 (the stream is not being consumed by another client).

**What this does not show:** that the Motive project was deleted or renumbered, or
that the cameras are off. Every observation is consistent with Motive running with
its Data Streaming pane switched off, or bound to a server-side interface that does
not reach this subnet. Distinguishing those needs a look at the Motive host's screen.

### Consequence for the ID dispute

Unresolved. Committed RS485 code says 500–505; the brief says 1000–1005 (RS485) and
2000–2005 (CAN); legacy VNEMA agrees with the brief on the RS485 arm. No id was
guessed into code.

## 2. CanArmMocap live — not run (blocked)

`hw_tests/canarm_mocap_live.py` refuses within 5 s on a dead stream, distinguishing
"no frames at all" from "frames that will not convert". Rehearsed end-to-end against
`sim_stream.CanArmSimStream` (which feeds the real receiver's listeners in-process):

```
mode=sim rb_id_base=2000 n_bodies=6
frames=961  valid_frames=961  fps=122.3  q_stale_fraction=0.000
per-body: 2000..2005 present=1563/1563, absent=0, unsolved=0, dropout_events=0
```

The per-body absence counters exist because `MocapRx` keeps a row's previous pose
when a frame omits it — a silently lost plate looks exactly like a still plate unless
something counts the absences.

## 3. Chain measurement — not run; `canarm_params.py` deliberately untouched

Placeholder chain still bit-identical to the RS485 nominal (total 0.692772 m). The
procedure is verified on the synthetic stream (recovers generator lengths to ~1e-15)
and the mechanical edit is documented in `canarm_params.py`'s docstring: LL from
intra-segment gaps minus (AA1+AA2)=0.057 m, JD from inter-segment gaps, then
`MEASURED = True` with provenance; `viz/self_check.py` re-verifies the drawn
geometry automatically.

## 4. Temporary marker locks — skipped

No frames, so no rest capture; and whether labeled markers are streamed is unknown
(the zeros are the zeros of a dead stream). `templates/canarm_locks.json` still does
not exist and `load_canarm_locks()` refuses by name.

## 5. Room roster — nothing observable

CAN block, RS485 block, and Kinova 1008 all absent for the same reason.

## 6. Offline state — green and undisturbed

```
pytest UMArm_KINEMATICS UMArm_MOCAP -q   568 passed, 6 skipped
viz/self_check.py                        ALL CHECKS PASSED
canarm_control_gui.py --self-test        OK
```

## Re-run sequence once Motive streams again

1. `hw_tests\mocap_wire_probe.py` — ping reply = server answers; datagrams = it streams.
2. `hw_tests\mocap_census.py --seconds 10` — read the contiguous-blocks line; if a
   block other than 2000–2005 is the CAN arm, change `CANARM_RB_ID_BASE` in
   `canarm_mocap.py` and nowhere else.
3. `hw_tests\canarm_mocap_live.py` — q health + chain gaps; then edit
   `canarm_params.py` per its docstring and re-run `pytest UMArm_KINEMATICS`,
   `viz\self_check.py`.
4. `UMArm_MOCAP\mocap_probe.py --rb-id-base 2000 --n-bodies 6 --lock-out
   UMArm_MOCAP\templates\canarm_locks.json` for the temporary marker locks; then
   compare `CanArmMarkerMocap` q vs streamed-pose q (a few degrees offset expected).
5. If the census still sees nothing while Motive claims to stream: run
   `hw_tests\mocap_sniff.py` elevated (`SIO_RCVALL`) to answer whether any packet
   from 192.168.1.100 reaches this NIC at all.
