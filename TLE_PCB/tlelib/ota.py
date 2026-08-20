"""Broadcast firmware update over CAN.

One image is streamed once, on the broadcast ID, and every selected board
writes it. That is the point: flashing eight boards costs the same wire time as
flashing one. Boards stay silent during the stream -- with eight of them
receiving, a per-frame acknowledgement would put eight frames on the bus for
every one the host sends -- so progress is read back with an explicit status
poll at each block boundary, and any board that fell behind is topped up over
its own unicast data ID before the next block starts.

A board's ID lives in the NVS partition, which no part of an OTA touches, so
flashing never renames a board.

The block geometry (128 frames of 7 payload bytes = 896 B per flash write) and
the START/END handshake are the ones the original firmware uses, so this driver
updates old boards and TLE boards from the same code path.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

from . import proto as P
from .canlink import CanLink
from .timing import hires_clock, sleep_until

# Gap between transmitted OTA frames. A 15-byte standard frame is about 130 us
# on the wire at 1 Mbit/s, so this leaves the receiving board roughly 3x
# headroom to pull each one out of the controller before the next arrives.
FRAME_GAP_S = 0.0004

# Frames per serial write, and the reason it is one.
#
# Batching several frames into a single write is the obvious throughput
# optimisation -- each write is a USB transaction, and transactions cost more
# than the wire does. It also breaks the transfer. Measured on this bench, one
# 896-byte block at a 300 us offered gap:
#
#     batch 8 -> board accepted 1 frame     batch 2 -> accepted 3
#     batch 4 -> board accepted 2 frames    batch 1 -> accepted all 128
#
# Frames written together are handed to the wire back to back, with only the
# inter-frame space between them. Broadcast image data all arrives on one CAN
# ID, which the receiving board filters into a single receive buffer, so it has
# one frame time to read each one out over SPI before the next overwrites it.
# Pacing per frame keeps it ahead; batching does not, at any gap, because the
# gap lands between batches rather than between frames.
BATCH_FRAMES = 1

START_ACK_TIMEOUT_S = 3.0
END_ACK_TIMEOUT_S = 5.0
STATUS_TIMEOUT_S = 1.0
DEFAULT_RETRIES = 5


class OtaCancelled(RuntimeError):
    pass


class BroadcastOta:
    def __init__(self, link: CanLink, log=print, progress=None,
                 cancel: threading.Event | None = None,
                 frame_gap_s: float = FRAME_GAP_S, batch_frames: int = BATCH_FRAMES):
        self.link = link
        self.log = log
        self._progress = progress
        self.cancel = cancel or threading.Event()
        self.frame_gap_s = frame_gap_s
        self.batch_frames = max(1, batch_frames)
        self.frames_sent = 0
        hires_clock()

    # ---- helpers -------------------------------------------------------
    def _check(self) -> None:
        if self.cancel.is_set():
            raise OtaCancelled("cancelled")

    def _report(self, base: int, state: str, fraction: float) -> None:
        if self._progress is not None:
            self._progress(base, state, max(0.0, min(1.0, fraction)))

    def _send_paced(self, frames: list[tuple[int, bytes]]) -> None:
        """Write frames in batches, paced to the configured inter-frame gap."""
        deadline = time.perf_counter()
        for start in range(0, len(frames), self.batch_frames):
            self._check()
            batch = frames[start:start + self.batch_frames]
            self.link.send_batch(batch)
            self.frames_sent += len(batch)
            deadline += self.frame_gap_s * len(batch)
            sleep_until(deadline)

    @staticmethod
    def _block_frames(block: bytes) -> list[bytes]:
        return [P.build_ota_data(seq, block[seq * P.OTA_DATA_PER_FRAME:
                                            (seq + 1) * P.OTA_DATA_PER_FRAME])
                for seq in range(math.ceil(len(block) / P.OTA_DATA_PER_FRAME))]

    @staticmethod
    def _block_accepted(status: P.OtaStatus | None, total_frames: int, block_len: int,
                        blocks_expected: int) -> bool:
        """Has this board taken the whole block?

        A full block leaves the node with expected_seq back at 0, one more block
        written; a short final block leaves the bytes buffered for END to flush.

        `blocks_expected` is what makes this safe. A board that heard *nothing*
        of the block reports exactly the same expected_seq 0 / buffer_index 0 as
        one that took all 128 frames and flashed them -- the sequence-error flag
        is only set when a wrongly-numbered frame actually arrives. Without the
        written-block counter the host would record a silently missed block as
        accepted, and every later block would be written at the wrong offset.
        """
        if status is None or not status.active or status.has_error:
            return False
        if status.blocks_written < blocks_expected:
            return False
        if total_frames >= P.OTA_FRAMES_PER_BLOCK:
            return status.expected_seq == 0 and status.buffer_index == 0
        return status.expected_seq == total_frames and status.buffer_index == block_len

    @staticmethod
    def _resume_sequence(status: P.OtaStatus | None, total_frames: int) -> int:
        """Where a lagging board wants the block resumed from."""
        if status is None:
            return 0
        if status.expected_seq >= total_frames:
            return 0
        return status.expected_seq

    # ---- phases --------------------------------------------------------
    def start_nodes(self, bases: list[int]) -> list[int]:
        """START each board (unicast, acknowledged). Returns those that failed."""
        failures: list[int] = []
        for base in bases:
            self._check()
            self._report(base, "starting", 0.0)
            self.link.drain()
            self.link.send(P.ids(base).ctrl, P.build_simple_command(P.CMD_START))
            ok, _ = self.link.wait_ack(base, START_ACK_TIMEOUT_S)
            if not ok:
                self.log(f"[ERROR] 0x{base:03X}: no START acknowledgement")
                self._report(base, "failed", 0.0)
                failures.append(base)
        return failures

    def repair(self, base: int, frames: list[bytes], start_sequence: int) -> None:
        self.log(f"0x{base:03X}: resending frames {start_sequence}..{len(frames) - 1}")
        data_id = P.ids(base).data
        self._send_paced([(data_id, f) for f in frames[start_sequence:]])
        if len(frames) >= P.OTA_FRAMES_PER_BLOCK:
            self.link.wait_ack(base, STATUS_TIMEOUT_S)

    def upload(self, image: bytes, bases: list[int], retries: int = DEFAULT_RETRIES) -> dict[int, bool]:
        """Stream `image` to every board in `bases`. Returns {base: succeeded}."""
        selected = list(dict.fromkeys(bases))
        if not selected:
            return {}
        if not image:
            raise ValueError("firmware image is empty")

        total_blocks = math.ceil(len(image) / P.OTA_BLOCK_BYTES)
        self.log(f"Broadcast update: {len(selected)} board(s) "
                 f"[{', '.join(f'0x{b:03X}' for b in selected)}], "
                 f"{len(image)} bytes, {total_blocks} blocks")

        result = {base: False for base in selected}
        failed = set(self.start_nodes(selected))
        # Every board that acknowledged START now has its control loop stopped,
        # its outputs off and its sync handling gated. Whatever happens after
        # this, each one has to be given an END or it stays like that.
        opened = [b for b in selected if b not in failed]
        active = list(opened)
        finalised: set[int] = set()
        if not active:
            return result

        try:
            started = time.perf_counter()
            for index in range(total_blocks):
                self._check()
                block = image[index * P.OTA_BLOCK_BYTES:(index + 1) * P.OTA_BLOCK_BYTES]
                frames = self._block_frames(block)
                blocks_expected = index + 1 if len(frames) >= P.OTA_FRAMES_PER_BLOCK else index

                self.link.drain()
                self._send_paced([(P.ID_BROADCAST, f) for f in frames])

                statuses = {b: self.link.ota_status(b, STATUS_TIMEOUT_S)
                            for b in active if b not in failed}
                for attempt in range(retries + 1):
                    pending = [b for b, st in statuses.items()
                               if not self._block_accepted(st, len(frames), len(block),
                                                           blocks_expected)]
                    if not pending:
                        break
                    if attempt == retries:
                        for base in pending:
                            self.log(f"[ERROR] 0x{base:03X}: gave up on block {index + 1}")
                            self._report(base, "failed", index / total_blocks)
                            failed.add(base)
                        break
                    for base in pending:
                        self._check()
                        if statuses[base] is None:
                            # The poll was lost. That says nothing about how much
                            # of the block the board took, and resending on the
                            # assumption it took none would append the same 896
                            # bytes to the image a second time. Ask again.
                            statuses[base] = self.link.ota_status(base, STATUS_TIMEOUT_S)
                            continue
                        self._report(base, "repair", index / total_blocks)
                        self.repair(base, frames,
                                    self._resume_sequence(statuses[base], len(frames)))
                        statuses[base] = self.link.ota_status(base, STATUS_TIMEOUT_S)

                for base in active:
                    if base not in failed:
                        self._report(base, "writing", (index + 1) / total_blocks)
                if len(failed) == len(selected):
                    self.log("[ERROR] every board failed; stopping")
                    return result

            elapsed = time.perf_counter() - started
            self.log(f"Image streamed in {elapsed:.1f} s "
                     f"({len(image) / max(elapsed, 1e-6) / 1024:.1f} kB/s on the wire)")

            for base in active:
                if base in failed:
                    continue
                self._check()
                self._report(base, "finalizing", 1.0)
                self.link.drain()
                self.link.send(P.ids(base).ctrl, P.build_simple_command(P.CMD_END))
                finalised.add(base)
                ok, _ = self.link.wait_ack(base, END_ACK_TIMEOUT_S)
                if ok:
                    result[base] = True
                    self._report(base, "done", 1.0)
                    self.log(f"0x{base:03X}: updated, rebooting into the new image")
                else:
                    self._report(base, "failed", 1.0)
                    self.log(f"[ERROR] 0x{base:03X}: no END acknowledgement")
            return result
        finally:
            self._close_sessions([b for b in opened if b not in finalised])

    def _close_sessions(self, bases: list[int]) -> None:
        """END every session left open, including after a failure or a cancel.

        This protocol has no ABORT, and a board with an open session has its
        control loop stopped, its outputs off and its sync handling gated -- it
        stays that way until something closes the session or it is power cycled.
        END is safe even for a partial image: the board fails validation,
        refuses to switch boot slots, and comes back on the one it is running.
        """
        for base in bases:
            try:
                self.link.send(P.ids(base).ctrl, P.build_simple_command(P.CMD_END))
                self.link.wait_ack(base, STATUS_TIMEOUT_S)
                self.log(f"0x{base:03X}: OTA session closed")
            except Exception as exc:
                self.log(f"[ERROR] 0x{base:03X}: could not close its OTA session: {exc}")


def image_project_name(image: bytes) -> str:
    """The ESP-IDF project name recorded in the image.

    An application image carries an `esp_app_desc_t` immediately after the
    image header and the first segment header: 24 + 8 = 32 bytes in, magic
    0xABCD5432, with `project_name` 48 bytes further on. It is the only thing
    in the file that says which firmware this is, and it is what stops a TLE
    image being broadcast into boards that run something else.
    """
    if len(image) < 0x80:
        return ""
    if int.from_bytes(image[0x20:0x24], "little") != 0xABCD5432:
        return ""
    name = image[0x50:0x70]
    end = name.find(0)
    if end >= 0:
        name = name[:end]
    return name.decode("ascii", errors="replace")


# ESP-IDF project name -> the variants a board may report and still be a valid
# target for that image.
#
# A set, not a single variant, because the old project builds both its variants
# from one source tree: 7 mm and DT differ only by a compile-time valve type, so
# the project name cannot tell them apart and refusing one of them would be
# wrong. What this must catch is the family boundary -- an image for one
# firmware reaching boards running the other, which is unrecoverable without
# physically reflashing every one of them over USB.
PROJECT_VARIANTS = {
    "VEMA_MAX22200": frozenset({P.VARIANT_TLE_DVP}),
    "Valve_not_embedded_XL": frozenset({P.VARIANT_7MM, P.VARIANT_DT}),
}


def image_variants(image: bytes) -> frozenset[int] | None:
    """Variants this image may be flashed onto, or None if it cannot be told."""
    return PROJECT_VARIANTS.get(image_project_name(image))


def image_variant(image: bytes) -> int | None:
    """The single variant this image targets, when there is exactly one."""
    variants = image_variants(image)
    if variants is not None and len(variants) == 1:
        return next(iter(variants))
    return None


def load_image(path: str | Path) -> bytes:
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"{path} is empty")
    # An ESP-IDF application image starts with the 0xE9 magic byte. Catching a
    # wrong file here is much cheaper than catching it after a board has erased
    # its spare slot.
    if data[0] != 0xE9:
        raise ValueError(f"{path} does not look like an ESP32 application image "
                         f"(first byte 0x{data[0]:02X}, expected 0xE9)")
    return data
