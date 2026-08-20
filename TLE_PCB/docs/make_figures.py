#!/usr/bin/env python3
"""Generate the report figures as standalone SVG, from code.

No plotting library: every figure is emitted as explicit SVG, so the report has
no build dependencies and each drawing says exactly what the text claims. The
numbers that appear inside the drawings are imported from `tlelib.proto` or
passed in from the measurement runs, so a figure cannot drift away from the
protocol it describes.

    python make_figures.py            # writes figures/*.svg next to this file
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tlelib import proto as P  # noqa: E402

OUT = HERE / "figures"

INK = "#1a1f2b"
MUTED = "#6b7280"
LINE = "#c3c9d4"
ACCENT = "#2563a8"
WARM = "#b4531f"
GOOD = "#2f7d55"
PLUM = "#6b3fa0"
FILL_A = "#e8eef7"
FILL_B = "#f6efe7"
FILL_C = "#eaf3ee"
FILL_D = "#f0ebf7"
MONO = "ui-monospace, 'Cascadia Mono', Consolas, monospace"
SANS = "Inter, 'Segoe UI', system-ui, sans-serif"


def esc(text_in: str) -> str:
    return text_in.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(width: int, height: int, body: str, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="100%" role="img" aria-label="{esc(title)}" '
            f'font-family="{SANS}">\n{body}\n</svg>\n')


def text(x, y, s, size=12, fill=INK, anchor="start", weight="400", family=SANS, opacity=1.0):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{family}" opacity="{opacity}">{esc(s)}</text>')


def rect(x, y, w, h, fill="#fff", stroke=LINE, width=1.2, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def box(x, y, w, h, label, sub="", fill=FILL_A, stroke=ACCENT, rx=6, size=12):
    out = rect(x, y, w, h, fill, stroke, 1.2, rx)
    if sub:
        out += text(x + w / 2, y + h / 2 - 3, label, size=size, weight="600", anchor="middle")
        out += text(x + w / 2, y + h / 2 + 11, sub, size=size - 2.5, fill=MUTED, anchor="middle")
    else:
        out += text(x + w / 2, y + h / 2 + 4, label, size=size, weight="600", anchor="middle")
    return out


def line(x1, y1, x2, y2, stroke=LINE, width=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}"{d}/>')


def arrow(x1, y1, x2, y2, stroke=ACCENT, width=1.4, dash=None, marker="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}" marker-end="url(#{marker})"{d}/>')


def path(d, stroke=ACCENT, width=2.0, fill="none", dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}/>')


def marker(name, colour):
    return (f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>')


DEFS = ("<defs>" + marker("ah", ACCENT) + marker("ahg", GOOD) + marker("ahw", WARM)
        + marker("ahm", MUTED) + marker("ahp", PLUM) + "</defs>")


# ------------------------------------------------------------------ figure 1
def fig_bus(latency_ms: dict) -> str:
    """The mixed bus: what is on it and which protocol layer answers."""
    W, H = 780, 258
    b = [DEFS]

    b.append(text(0, 12, "Twenty-four actuators, two board types, one protocol",
                  size=11.5, fill=MUTED))

    # host
    b.append(box(8, 34, 150, 62, "PC backend", "sync master, 150 Hz", fill=FILL_B, stroke=WARM))
    b.append(box(8, 100, 150, 34, "CANable 2.0", "slcan, 1 Mbit/s", fill="#fbfbfd", stroke=LINE))
    b.append(arrow(83, 96, 83, 100, stroke=WARM))

    # the wire
    b.append(line(20, 152, 760, 152, stroke=INK, width=2.6))
    # Left of the first drop line, so nothing crosses the label.
    b.append(text(20, 147, "CAN  1 Mbit/s", size=9.5, fill=MUTED, family=MONO))
    b.append(arrow(83, 134, 83, 150, stroke=WARM))

    # TLE boards
    b.append(rect(190, 178, 268, 74, FILL_A, ACCENT, 1.4, 8))
    b.append(text(200, 195, "8 x TLE board  (new)", size=11, weight="700", fill=ACCENT))
    b.append(text(200, 211, "0x101 - 0x108   proportional, Clippard DVP",
                  size=9.5, fill=INK, family=MONO))
    b.append(text(200, 226, "table groups 0x091 - 0x093", size=9.5, fill=MUTED, family=MONO))
    b.append(text(200, 242, "replaces the eight DT big-valve actuators", size=9.5, fill=MUTED))
    for i in range(8):
        b.append(line(206 + i * 32, 152, 206 + i * 32, 178, stroke=ACCENT, width=1.2))

    # legacy boards
    b.append(rect(486, 178, 274, 74, FILL_C, GOOD, 1.4, 8))
    b.append(text(496, 195, "16 x VNEMA_MK8 board  (existing)", size=11, weight="700", fill=GOOD))
    b.append(text(496, 211, "0x109 - 0x118   on/off, Clippard 7 mm",
                  size=9.5, fill=INK, family=MONO))
    b.append(text(496, 226, "table groups 0x093 - 0x098", size=9.5, fill=MUTED, family=MONO))
    b.append(text(496, 242, "firmware unchanged for this work", size=9.5, fill=MUTED))
    for i in range(16):
        b.append(line(494 + i * 17, 152, 494 + i * 17, 178, stroke=GOOD, width=1.0))

    # what the host puts on the wire each cycle
    b.append(rect(190, 30, 570, 110, "#fff", LINE, 1.2, 8))
    b.append(text(202, 48, "every 6.67 ms the host sends", size=10.5, weight="700", fill=INK))
    b.append(box(202, 58, 128, 32, "0x091..0x098", "8 target frames", fill=FILL_B,
                 stroke=WARM, size=10))
    b.append(box(344, 58, 104, 32, "0x090  DLC 0", "the sync edge", fill=FILL_B,
                 stroke=WARM, size=10))
    b.append(arrow(330, 74, 344, 74, stroke=WARM))
    b.append(text(202, 110, "and every addressed board answers on its own ID with a 2-byte status",
                  size=9.8, fill=MUTED))
    b.append(text(202, 126, "pressure latched at the edge, so one host row is one instant "
                            "across all 24", size=9.8, fill=MUTED))
    b.append(box(468, 58, 130, 32, "reply  DLC 2", "12-bit + 4 flags", fill=FILL_A,
                 stroke=ACCENT, size=10))
    b.append(text(612, 70, f"{latency_ms['p50']:.2f} ms median", size=9.5, fill=ACCENT,
                  family=MONO, weight="600"))
    b.append(text(612, 84, f"{latency_ms['max']:.2f} ms worst", size=9.5, fill=MUTED, family=MONO))

    return svg(W, H, "\n".join(b), "The shared CAN bus and its two board types")


# ------------------------------------------------------------------ figure 2
def fig_cores() -> str:
    """Task and core map, and the two SPI buses that force the split."""
    W, H = 780, 290
    b = [DEFS]

    b.append(text(0, 12, "One 6 kHz timer is the only clock; the two cores split the bus "
                         "from the plant", size=11.5, fill=MUTED))

    # timer spine
    b.append(rect(0, 20, W, 26, FILL_B, WARM, 1.2, 5))
    b.append(text(12, 38, "GPTimer base tick  6000 Hz", size=12, weight="700", fill=WARM))
    for offset, label in ((236, "/2 -> ADC 3 kHz"), (392, "/4 -> MAX update 1.5 kHz"),
                          (596, "/8 -> PID 750 Hz")):
        b.append(text(offset, 38, label, size=10.5, fill=INK, family=MONO))

    # core 0
    b.append(rect(8, 62, 372, 158, "#fff", LINE, 1.2, 8))
    b.append(text(20, 79, "core 0   bus and control law", size=11, weight="700", fill=MUTED))
    b.append(text(370, 79, "SPI2: MCP25625 alone", size=9, fill=MUTED, family=MONO,
                  anchor="end"))
    b.append(box(18, 84, 352, 32, "can   prio 7", "MCP INT -> drain RX -> route -> TX slot"))
    b.append(box(18, 120, 352, 32, "pid   prio 6", "750 Hz, consumes the ADC window"))
    b.append(box(18, 156, 352, 26, "usb   prio 3", "text frames on USB-Serial/JTAG",
                 fill="#fbfbfd", stroke=LINE, size=10.5))
    b.append(box(18, 186, 352, 26, "rawdump  prio 2", "ADC capture dump, below everything",
                 fill="#fbfbfd", stroke=LINE, size=10.5))

    # core 1
    b.append(rect(400, 62, 372, 128, "#fff", LINE, 1.2, 8))
    b.append(text(412, 79, "core 1   sensing and actuation", size=11, weight="700", fill=MUTED))
    b.append(box(410, 84, 352, 32, "adc   prio 6", "3 kHz LTC1864, outlier-rejected",
                 fill=FILL_C, stroke=GOOD))
    b.append(box(410, 120, 352, 32, "valve prio 5", "1.5 kHz TLE92464 current update",
                 fill=FILL_C, stroke=GOOD))
    b.append(text(412, 174, "SPI3: TLE92464 + LTC1864 share MOSI/SCK/MISO",
                  size=9, fill=MUTED, family=MONO))

    # cross-core data flow
    b.append(arrow(380, 100, 410, 100, stroke=MUTED, marker="ahm"))
    b.append(arrow(410, 136, 380, 136, stroke=MUTED, marker="ahm"))
    b.append(text(395, 92, "target", size=9, fill=MUTED, anchor="middle"))
    b.append(text(395, 152, "samples", size=9, fill=MUTED, anchor="middle"))

    # the reason for the split
    b.append(rect(8, 230, 764, 54, FILL_D, PLUM, 1.2, 8))
    b.append(text(20, 247, "Why the split is this way", size=10.5, weight="700", fill=PLUM))
    b.append(text(20, 262, "Separate SPI buses, so CAN traffic and valve current updates never "
                           "queue behind each other.", size=9.6, fill=INK))
    b.append(text(20, 276, "CAN outranks the control law: the sync edge defines when the whole "
                           "robot changes target, so servicing it late is worse than running "
                           "the loop late.", size=9.6, fill=INK))

    return svg(W, H, "\n".join(b), "ESP32-S3 task and core map")


# ------------------------------------------------------------------ figure 3
def fig_cycle(latency_ms: dict) -> str:
    """One 150 Hz cycle, drawn to scale, with measured reply latency."""
    W, H = 780, 268
    period_ms = 1000.0 / P.CYCLE_HZ
    x0, x1 = 92, 736
    scale = (x1 - x0) / period_ms

    def at(ms: float) -> float:
        return x0 + ms * scale

    b = [DEFS]
    b.append(text(0, 13, f"One cycle drawn to scale: {period_ms:.2f} ms at "
                         f"{P.CYCLE_HZ:.0f} Hz", size=11.5, fill=MUTED))

    # lanes
    lanes = (("host TX", 46, WARM, FILL_B), ("wire", 106, INK, "#fff"),
             ("board", 166, ACCENT, FILL_A))
    for label, y, colour, fill in lanes:
        b.append(rect(x0, y, x1 - x0, 34, fill, LINE, 1.0, 4))
        b.append(text(x0 - 10, y + 22, label, size=10, fill=colour, anchor="end", weight="600"))

    # time axis
    b.append(line(x0, 214, x1, 214, stroke=LINE))
    for ms in range(0, 7):
        b.append(line(at(ms), 214, at(ms), 219, stroke=LINE))
        b.append(text(at(ms), 232, f"{ms}", size=9, fill=MUTED, anchor="middle", family=MONO))
    b.append(text(at(period_ms), 232, f"{period_ms:.2f}", size=9, fill=MUTED,
                  anchor="middle", family=MONO))
    b.append(text((x0 + x1) / 2, 250, "milliseconds after the cycle starts", size=9.5,
                  fill=MUTED, anchor="middle"))

    # host transmit burst: 8 table frames then the sync
    for i in range(8):
        b.append(rect(at(0.02 + i * 0.136), 52, 0.12 * scale, 22, FILL_B, WARM, 1.0, 2))
    b.append(rect(at(1.11), 52, 0.06 * scale, 22, "#f7d9c0", WARM, 1.2, 2))
    b.append(text(at(0.55), 44, "8 target frames", size=9.5, fill=WARM, anchor="middle"))
    b.append(text(at(1.14), 44, "sync", size=9.5, fill=WARM, weight="700"))

    # the sync edge
    b.append(line(at(1.14), 30, at(1.14), 214, stroke=WARM, width=1.6, dash="4 3"))

    # board lane
    b.append(rect(at(1.14), 172, 0.35 * scale, 22, "#d8e5f4", ACCENT, 1.2, 2))
    b.append(text(at(1.55), 187, "latch pressure, promote target", size=9.5, fill=ACCENT))

    p50, p95, worst = latency_ms["p50"], latency_ms["p95"], latency_ms["max"]
    b.append(rect(at(p50) - 3, 112, 6, 22, "#cfe0f2", ACCENT, 1.2, 2))
    b.append(line(at(p50), 106, at(worst), 106, stroke=ACCENT, width=1.0))
    b.append(line(at(worst), 100, at(worst), 112, stroke=ACCENT, width=1.0))
    b.append(text(at(p50) + 10, 128, f"reply  p50 {p50:.2f}  p95 {p95:.2f}  max {worst:.2f} ms",
                  size=9.5, fill=ACCENT, family=MONO))

    # receive window
    window = period_ms * 0.82
    b.append(rect(at(0), 196, window * scale, 8, "#eef3fa", ACCENT, 0.8, 2))
    b.append(text(at(window) + 6, 204, "host receive window (82 %)", size=9, fill=MUTED))

    # margin
    b.append(rect(at(window), 46, (period_ms - window) * scale, 158, "#fbf3ee", WARM, 0.8, 3,
                  dash="3 3"))
    b.append(text(at(window) + 4, 40, "margin", size=9, fill=WARM))

    return svg(W, H, "\n".join(b), "The 150 Hz cycle, to scale")


# ------------------------------------------------------------------ figure 4
def fig_protocol() -> str:
    """Identifier map and the frame layouts that matter."""
    W, H = 780, 296
    b = [DEFS]

    b.append(text(0, 12, "Identifier map, and the two frames that carry the control loop",
                  size=11.5, fill=MUTED))

    rows = [
        ("0x090", "broadcast", "DLC 0 is the sync edge. DLC > 0 is OTA image data.", WARM, FILL_B),
        ("0x091 - 0x098", "target table", "Three actuators per frame, so 24 targets cost 8 frames.", WARM, FILL_B),
        ("0x101 - 0x118", "actuator base", "Direct command in, compact status out.", ACCENT, FILL_A),
        ("base + 0x100", "host control", "OTA start/end, set ID, diagnostics, version.", ACCENT, FILL_A),
        ("base + 0x200", "unicast data", "Repairs one board that fell behind the broadcast.", ACCENT, FILL_A),
        ("base + 0x300", "board status", "Acknowledgements, diagnostics, extended telemetry.", ACCENT, FILL_A),
        ("base + 0x400", "extended command", "TLE only: gains, shaping, current codes. New.", PLUM, FILL_D),
    ]
    y = 26
    for ident, name, note, colour, fill in rows:
        b.append(rect(0, y, 300, 24, fill, colour, 1.0, 4))
        b.append(text(10, y + 16, ident, size=10, family=MONO, weight="700", fill=colour))
        b.append(text(112, y + 16, name, size=10, fill=INK))
        b.append(text(312, y + 16, note, size=9.6, fill=MUTED))
        y += 27

    # frame layouts
    def field(x, y, w, label, sub, fill, stroke):
        out = rect(x, y, w, 32, fill, stroke, 1.1, 3)
        out += text(x + w / 2, y + 14, label, size=9.5, anchor="middle", weight="700",
                    family=MONO)
        out += text(x + w / 2, y + 26, sub, size=8.2, anchor="middle", fill=MUTED)
        return out

    b.append(text(0, 236, "compact command / status   DLC 2, little-endian", size=10,
                  weight="700", fill=ACCENT))
    b.append(field(0, 242, 216, "bits 0..11", "pressure, 12-bit counts", FILL_A, ACCENT))
    b.append(field(220, 242, 108, "bits 12..15", "flags", "#d8e5f4", ACCENT))
    b.append(text(0, 288, f"one count = {1 / P.TLE_COUNTS_PER_PSI:.4f} psi on a TLE board "
                          f"(raw >> {P.TLE_RAW_SHIFT})", size=9, fill=MUTED, family=MONO))

    b.append(text(400, 236, "target table   DLC 8", size=10, weight="700", fill=WARM))
    b.append(field(400, 242, 74, "byte 0", "0x80 | slot", FILL_B, WARM))
    b.append(field(478, 242, 58, "byte 1", "3-bit mask", FILL_B, WARM))
    b.append(field(540, 242, 76, "2..3", "slot + 0", "#f7e6d6", WARM))
    b.append(field(620, 242, 76, "4..5", "slot + 1", "#f7e6d6", WARM))
    b.append(field(700, 242, 76, "6..7", "slot + 2", "#f7e6d6", WARM))
    b.append(text(400, 282, "bit 7 of byte 0 separates a table frame from OTA data,",
                  size=9, fill=MUTED))
    b.append(text(400, 294, "whose first byte is a sequence number 0..127", size=9, fill=MUTED))

    return svg(W, H, "\n".join(b), "CAN identifier map and frame layouts")


# ------------------------------------------------------------------ figure 5
def fig_controller() -> str:
    """The proportional controller, with every feature on the signal path."""
    W, H = 780, 340
    b = [DEFS]

    b.append(text(0, 13, "The proportional loop: 750 Hz, fixed point, and what each stage is for",
                  size=11.5, fill=MUTED))

    # forward path
    b.append(box(0, 34, 96, 44, "target", "12-bit -> raw", fill=FILL_B, stroke=WARM, size=11))
    b.append(box(112, 34, 92, 44, "error", "raw counts", fill="#fff", stroke=LINE, size=11))
    b.append(arrow(96, 56, 112, 56))

    b.append(box(220, 34, 116, 44, "gain schedule", "kp x 19963 / raw", size=11))
    b.append(arrow(204, 56, 220, 56))
    b.append(box(352, 34, 104, 44, "soft zone", "P and I fade in", size=11))
    b.append(arrow(336, 56, 352, 56))
    b.append(box(472, 34, 104, 44, "D + D-fade", "damps approach", size=11))
    b.append(arrow(456, 56, 472, 56))
    b.append(box(592, 34, 104, 44, "I + anti-windup", "+-2x full scale", size=11))
    b.append(arrow(576, 56, 592, 56))

    # second row
    b.append(box(220, 116, 116, 44, "feed-forward", "leak make-up seed", fill=FILL_D,
                 stroke=PLUM, size=11))
    b.append(box(352, 116, 104, 44, "flow shaping", "scales the seed", fill=FILL_D,
                 stroke=PLUM, size=11))
    b.append(box(472, 116, 104, 44, "sum -> permille", "-1000 .. +1000", fill="#fff",
                 stroke=LINE, size=11))
    b.append(arrow(336, 138, 352, 138, stroke=PLUM, marker="ahp"))
    b.append(arrow(456, 138, 472, 138, stroke=PLUM, marker="ahp"))
    b.append(arrow(644, 78, 644, 100, stroke=ACCENT))
    b.append(path(f"M 644 100 L 644 108 L 524 108 L 524 116", stroke=ACCENT, width=1.4))

    # output stage
    b.append(box(0, 196, 148, 48, "sign picks the valve", "inlet ch2 / outlet ch3",
                 fill=FILL_C, stroke=GOOD, size=11))
    b.append(box(164, 196, 128, 48, "asymmetric map", "own open/max per valve",
                 fill=FILL_C, stroke=GOOD, size=11))
    b.append(box(308, 196, 116, 48, "slew limit", "4 codes per step",
                 fill=FILL_C, stroke=GOOD, size=11))
    b.append(box(440, 196, 128, 48, "output deadband", "quiet hold", fill=FILL_C,
                 stroke=GOOD, size=11))
    b.append(box(584, 196, 188, 48, "current code -> TLE92464", "I = code x 200/127 mA, "
                 "host ceiling 116", fill=FILL_C, stroke=GOOD, size=11))
    for x1, x2 in ((148, 164), (292, 308), (424, 440), (568, 584)):
        b.append(arrow(x1, 220, x2, 220, stroke=GOOD, marker="ahg"))
    b.append(path("M 524 160 L 524 176 L 74 176 L 74 196", stroke=ACCENT, width=1.4,
                  fill="none"))
    b.append(f'<polygon points="74,196 70,188 78,188" fill="{ACCENT}"/>')

    # feedback
    b.append(box(584, 274, 188, 44, "LTC1864  3 kHz", "trimmed window + EMA", fill="#fbfbfd",
                 stroke=LINE, size=11))
    b.append(path("M 584 296 L 158 296 L 158 78", stroke=MUTED, width=1.3, dash="4 3"))
    b.append(f'<polygon points="158,78 154,86 162,86" fill="{MUTED}"/>')
    b.append(text(320, 290, "measured pressure, 972.5 counts per psi", size=9.5, fill=MUTED))

    # tuned values
    b.append(rect(0, 262, 560, 62, "#fbfbfd", LINE, 1.0, 6))
    b.append(text(12, 280, "bench-tuned defaults, carried in the firmware", size=10,
                  weight="700", fill=INK))
    b.append(text(12, 297, "kp 2048   ki 2500   kd 2500   open 51   max 110   slew 4   "
                           "ff_scale 30   fs_seed 17", size=9.5, family=MONO, fill=INK))
    b.append(text(12, 313, "hardware dither off: in-loop it put a 150 Hz line 27x the noise "
                           "floor into the pressure", size=9.3, fill=MUTED))

    return svg(W, H, "\n".join(b), "The proportional pressure controller")


# ------------------------------------------------------------------ figure 6
def fig_ota(measured: dict) -> str:
    """Broadcast update: who says what, and why nobody acknowledges the data."""
    W, H = 780, 202
    b = [DEFS]

    b.append(text(0, 11, "One image on the wire, every board writing it", size=11.5, fill=MUTED))

    def stage(x, w, label, sub):
        out = rect(x, 28, w, 24, FILL_B, WARM, 1.1, 4)
        out += text(x + w / 2, 44, label, size=10, anchor="middle", weight="700", family=MONO)
        out += text(x + w / 2, 23, sub, size=8.6, anchor="middle", fill=MUTED)
        return out

    b.append(text(88, 44, "host", size=10, fill=WARM, anchor="end", weight="600"))
    b.append(stage(96, 108, "START x N", "unicast"))
    b.append(stage(214, 300, "128 data frames on 0x090", "broadcast, one per write"))
    b.append(stage(526, 128, "status poll x N", "unicast"))
    b.append(stage(664, 98, "END x N", "unicast"))

    # per-board reactions, three lanes standing for eight
    for label, y in (("board A", 70), ("board B", 96), ("board H", 122)):
        b.append(text(88, y + 13, label, size=10, fill=ACCENT, anchor="end", weight="600"))
        b.append(rect(96, y, 108, 18, FILL_A, ACCENT, 1.0, 3))
        b.append(text(150, y + 13, "ACK", size=9, anchor="middle", family=MONO, fill=ACCENT))
        b.append(rect(214, y, 300, 18, "#eef3fa", ACCENT, 1.0, 3))
        b.append(text(364, y + 13, "receive, no reply at all", size=9, anchor="middle",
                      fill=MUTED))
        b.append(rect(526, y, 128, 18, FILL_A, ACCENT, 1.0, 3))
        b.append(text(590, y + 13, "expected_seq", size=9, anchor="middle", family=MONO,
                      fill=ACCENT))
        b.append(rect(664, y, 98, 18, FILL_A, ACCENT, 1.0, 3))
        b.append(text(713, y + 13, "ACK, reboot", size=9, anchor="middle", fill=ACCENT))
    b.append(text(88, 112, "...", size=11, fill=MUTED, anchor="end"))

    b.append(rect(0, 152, 380, 30, FILL_B, WARM, 1.0, 5))
    b.append(text(10, 171, "silence during the stream is the point: 8 boards x 1 reply "
                           "per frame would be 8x the traffic", size=9.2, fill=INK))

    b.append(rect(392, 152, 388, 30, FILL_C, GOOD, 1.0, 5))
    b.append(text(402, 171, f"measured: {measured['bytes']} bytes, "
                            f"{measured['blocks']} blocks, {measured['seconds']:.1f} s, "
                            f"{measured['kbps']:.1f} kB/s, {measured['repairs']} repairs",
                  size=9.2, fill=INK, family=MONO))

    return svg(W, H, "\n".join(b), "Broadcast OTA sequence")


# ---------------------------------------------------------------------------
# Measurements, from the bench runs recorded in TLE_can_report.html section 6.
LATENCY_MS = {"min": 0.76, "p50": 1.07, "p95": 1.30, "p99": 1.37, "max": 1.91}
OTA_MEASURED = {"bytes": 324528, "blocks": 363, "seconds": 21.9, "kbps": 14.5, "repairs": 0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "bus.svg": fig_bus(LATENCY_MS),
        "cores.svg": fig_cores(),
        "cycle.svg": fig_cycle(LATENCY_MS),
        "protocol.svg": fig_protocol(),
        "controller.svg": fig_controller(),
        "ota.svg": fig_ota(OTA_MEASURED),
    }
    print("bus load, 24 boards @ 1 Mbit/s: "
          f"{P.bus_load(8, 16, 1_000_000, stuffing=False) * 100:.0f}-"
          f"{P.bus_load(8, 16, 1_000_000) * 100:.0f} %  (report section 4 table)")
    for name, body in figures.items():
        (OUT / name).write_text(body, encoding="utf-8")
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
