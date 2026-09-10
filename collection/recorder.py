"""The 150 Hz recording: one JSON object per sync edge, chunked to disk.

The schema is the one documented in ``digital_twin/data_schema.md``, adopted
as-is with the ``board_type`` addition that document specifies.  Three
properties of that schema are what make it worth adopting rather than
inventing:

* everything is stored **raw** — ADC counts and radians — with the per-id
  calibration snapshotted once, so a recalibration does not invalidate a
  recording;
* ``can_sync_time_s`` is stamped per cycle, which is what lets a rollout be
  aligned to the metal rather than to a nominal 6.667 ms grid;
* the mocap block sits **in the same object** as the joint state, with its own
  age, so the two halves of a sample are known to be the same instant rather
  than assumed to be.

Threading.  The whole point of :meth:`Recorder.on_cycle` is that it runs on the
CAN cycle thread, inside a 6.67 ms budget, and therefore does nothing but read
two already-held references and append one tuple to a deque.  All the
formatting and every write happens on a writer thread.  A recorder that
serialised JSON on the cycle thread would be measurable on the bus: the arm's
own operator GUI cost 15 Hz of cycle rate by redrawing a plot, and that was at
4 Hz, not 150.

Chunking.  A session is written as ``samples_chunk_NNNN_<reason>_<ts>.jsonl``
beside a ``metadata.json`` and a ``manifest.json``, so an interrupted session
leaves everything up to the last checkpoint readable.  ``manifest.json`` is
rewritten at every checkpoint, and its ``active`` flag stays true until a clean
stop — a session whose manifest still says ``active`` was interrupted, and its
last chunk is the short one.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque

import numpy as np

from UMArm_MOCAP.mocap_constants import U_JOINT_INDICES

SCHEMA_VERSION = 2

#: Cycles per chunk file.  9000 at 150 Hz is 60 s, which is the legacy sessions'
#: cadence and is short enough that an interrupted session loses at most a
#: minute of otherwise-good data.
CHUNK_CYCLES = 9000

#: How deep the hand-off deque may get before the recorder starts counting
#: drops.  At 150 Hz a 4000-entry queue is 27 s of backlog: far more than a
#: writer thread should ever need, so reaching it means the disk stalled, and
#: that is worth recording as a number rather than absorbing silently.
QUEUE_LIMIT = 4000


class Recorder:
    """Collects one row per sync edge and writes the session to disk.

    Constructed before the cycle starts, installed with
    ``backend.set_cycle_observer(rec.on_cycle)``, and closed after the cycle
    stops.  ``close()`` is idempotent and flushes.
    """

    def __init__(self, session_dir: str, *, ids, board_type, cals,
                 mocap=None, metadata=None, chunk_cycles: int = CHUNK_CYCLES):
        self.dir = session_dir
        os.makedirs(self.dir, exist_ok=True)
        self.ids = tuple(int(i) for i in ids)
        self.n = len(self.ids)
        self.board_type = [int(b) for b in board_type]
        #: ``{base: NodeCal}`` -- snapshotted so the reader can convert counts
        #: to Pa with the calibration that was in force at record time.
        self.cals = cals
        self.mocap = mocap
        self.chunk_cycles = int(chunk_cycles)

        self._q: deque = deque()
        self._qlock = threading.Lock()
        self._stop = threading.Event()
        self._writer = None
        self._fh = None
        self._chunk_index = 0
        self._chunk_rows = 0
        self._chunks: list = []
        self._chunk_start_cycle = 0
        self._chunk_start_t = None

        self.cycles = 0
        self.rows_written = 0
        self.dropped = 0
        self.t0_perf = None
        self.t0_wall = None
        #: Set by the campaign as it walks its phases; copied into every row so
        #: the training split can group by whole episode without a second file.
        self.segment = ""
        self.segment_kind = ""
        self.segment_index = -1

        self._meta = dict(metadata or {})
        self._last_mocap_seq = -1

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        self.t0_perf = time.perf_counter()
        self.t0_wall = time.time()
        self._write_metadata()
        self._stop.clear()
        self._writer = threading.Thread(target=self._write_loop,
                                        name="rec-writer", daemon=True)
        self._writer.start()

    def close(self) -> None:
        if self._writer is None:
            return
        self._stop.set()
        self._writer.join(timeout=60.0)
        self._writer = None
        self._close_chunk("stop")
        self._write_manifest(active=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- the cycle thread --------------------------------------------------- #

    def on_cycle(self, t_sync, targets, replies) -> None:
        """Installed as ``Backend``'s per-cycle observer.  Must stay cheap.

        Reads the mocap receiver's newest sample by reference rather than by
        copy — see :meth:`MocapRx.latest_sample`, which documents why that is
        safe — and appends one tuple.  No JSON, no formatting, no file I/O.
        """
        self.cycles += 1
        counts = [0] * self.n
        flags = [0] * self.n
        lat = [float("nan")] * self.n
        tgt = [0] * self.n
        ena = [0] * self.n
        for k, base in enumerate(self.ids):
            entry = targets.get(base)
            if entry is not None:
                tgt[k] = int(entry[0])
                ena[k] = int(bool(entry[1]))
            rep = replies.get(base)
            if rep is not None:
                t_reply, status = rep
                counts[k] = int(status.counts)
                flags[k] = int(status.flags)
                lat[k] = (t_reply - t_sync) * 1000.0

        mc = None
        if self.mocap is not None:
            mc = self.mocap.latest_sample()

        row = (t_sync, time.time(), tuple(counts), tuple(flags), tuple(lat),
               tuple(tgt), tuple(ena), len(replies), mc,
               self.segment, self.segment_kind, self.segment_index)
        with self._qlock:
            if len(self._q) >= QUEUE_LIMIT:
                self.dropped += 1
                return
            self._q.append(row)

    # -- the writer thread -------------------------------------------------- #

    def _write_loop(self) -> None:
        while True:
            with self._qlock:
                batch = list(self._q)
                self._q.clear()
            if batch:
                for row in batch:
                    self._emit(row)
            elif self._stop.is_set():
                return
            else:
                time.sleep(0.01)

    def _emit(self, row) -> None:
        (t_sync, t_wall, counts, flags, lat, tgt, ena, n_replied, mc,
         seg, seg_kind, seg_idx) = row
        if self._fh is None:
            self._open_chunk(t_sync)

        mocap_block = {"valid": False, "q_stale": True}
        if mc is not None and mc[0] is not None:
            t_mono, frame_no, q, homos, seq = mc
            fresh = seq != self._last_mocap_seq
            self._last_mocap_seq = seq
            # perf_counter and monotonic are different clocks on Windows; the
            # receiver stamps monotonic and the cycle stamps perf_counter, so
            # the age is taken against monotonic here and the offset between
            # the two is recorded once in metadata rather than per row.
            age_ms = (time.monotonic() - t_mono) * 1000.0
            centres = (homos[list(U_JOINT_INDICES), 0:3, 3]
                       if homos is not None else None)
            mocap_block = {
                "valid": True,
                "q_stale": bool(age_ms > 250.0),
                "fresh": bool(fresh),
                "frame": int(frame_no) if frame_no is not None else -1,
                "age_ms": round(age_ms, 3),
                "q": [round(float(v), 7) for v in q],
                "body_centers": ([round(float(v), 6) for v in centres.ravel()]
                                 if centres is not None else None),
            }

        obj = {
            "schema_version": SCHEMA_VERSION,
            "cycle": self.rows_written,
            "segment": seg,
            "segment_kind": seg_kind,
            "segment_index": seg_idx,
            "timestamp_unix_s": round(t_wall, 6),
            "can_sync_time_s": round(t_sync - self.t0_perf, 6),
            "robot_state": {"pressure_adc": list(counts)},
            "input": {"target_adc": list(tgt), "enabled": list(ena)},
            "actuator_status": list(flags),
            "actuator_reply_latency_ms": [None if v != v else round(v, 3)
                                          for v in lat],
            "cycle_responded": int(n_replied),
            "cycle_expected": self.n,
            "mocap": mocap_block,
        }
        self._fh.write(json.dumps(obj, separators=(",", ":")))
        self._fh.write("\n")
        self.rows_written += 1
        self._chunk_rows += 1
        if self._chunk_rows >= self.chunk_cycles:
            self._close_chunk("checkpoint")
            self._write_manifest(active=True)

    # -- chunk files -------------------------------------------------------- #

    def _open_chunk(self, t_sync) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"samples_chunk_{self._chunk_index:04d}_open_{ts}.jsonl"
        self._path = os.path.join(self.dir, name)
        self._fh = open(self._path, "w", encoding="utf-8", buffering=1 << 20)
        self._chunk_rows = 0
        self._chunk_start_cycle = self.rows_written
        self._chunk_start_t = t_sync - self.t0_perf

    def _close_chunk(self, reason: str) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        final = self._path.replace("_open_", f"_{reason}_")
        try:
            os.replace(self._path, final)
        except OSError:
            final = self._path
        self._chunks.append({
            "path": os.path.basename(final),
            "reason": reason,
            "samples": self._chunk_rows,
            "start_cycle": self._chunk_start_cycle,
            "end_cycle": self.rows_written,
            "start_time_s": self._chunk_start_t,
        })
        self._chunk_index += 1

    # -- the two index files ------------------------------------------------ #

    def _write_metadata(self) -> None:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "created_at_unix_s": self.t0_wall,
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_rate_hz": 150.0,
            "state_fields": ["pressure_adc", "q"],
            "input_fields": ["target_adc", "enabled"],
            "pressure_units": "adc_counts",
            "selected_ids": [f"0x{b:03X}" for b in self.ids],
            "board_type": self.board_type,
            "board_type_meaning": {"0": "7mm", "1": "DT/big-valve", "2": "TLE/DVP"},
            "node_cal": {
                f"0x{b:03X}": {"zero_counts": float(self.cals[b].zero_counts),
                               "counts_per_psi": float(self.cals[b].counts_per_psi)}
                for b in self.ids if b in self.cals},
            "sync_semantics": (
                "can_sync_time_s is seconds since the recorder's t0 on the "
                "host's perf_counter, stamped immediately before the runtime "
                "table and the sync edge are written to the bus. Every board "
                "latches its filtered pressure and promotes its staged target "
                "at that edge, so it is the one instant the whole arm agrees "
                "on and the only correct anchor for a rollout. pressure_adc "
                "in a row is the value the board reported in ITS reply to "
                "THIS edge, and target_adc is what this edge carried -- so the "
                "reported pressure is a response to the PREVIOUS edge's target."),
            "clock_note": (
                "can_sync_time_s comes from time.perf_counter and mocap ages "
                "from time.monotonic; on Windows these are different clocks. "
                "perf_minus_monotonic_s below is their offset, measured once "
                "at session start."),
            "perf_minus_monotonic_s": time.perf_counter() - time.monotonic(),
        }
        meta.update(self._meta)
        with open(os.path.join(self.dir, "metadata.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _write_manifest(self, *, active: bool) -> None:
        man = {
            "total_samples": self.rows_written,
            "checkpoint_count": len(self._chunks),
            "active": bool(active),
            "cycles_observed": self.cycles,
            "rows_dropped": self.dropped,
            "chunks": self._chunks,
        }
        with open(os.path.join(self.dir, "manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(man, fh, indent=2)
