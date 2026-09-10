"""The deliverable video's captions: one glyph run per caption, no doubled tail.

Offline and fast: draws onto a numpy canvas, opens no video and no GL context.
What is pinned is the defect the 2026-09-10 frame showed -- "the the recorded
pressure targetsts", "worst 13.43 degeg" -- which came from the dark halo pass
being drawn with a thicker stroke that, under OpenCV 5.0, also advances every
glyph further than the light pass on top of it.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from digital_twin import side_by_side as SBS  # noqa: E402

GREY = 128


def _extent(mask):
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols.min()), int(cols.max())) if cols.size else None


@pytest.mark.parametrize("text,scale", [
    ("open loop, held-out validation; the twin sees only the", 0.42),
    ("t  35.00 s   mean |q err|  4.41 deg   worst 13.43 deg", 0.5),
    ("TWIN (predicted)", 0.6),
])
def test_the_halo_never_runs_past_the_caption_it_outlines(text, scale):
    canvas = np.full((80, 1000, 3), GREY, dtype=np.uint8)
    SBS._label(cv2, canvas, text, (20, 50), scale=scale)
    light = _extent((canvas > 200).all(axis=2))
    dark = _extent((canvas < 40).all(axis=2))
    assert light is not None and dark is not None
    # The halo may reach one stroke offset (plus antialiasing) past the light
    # glyphs on either side, and no further.
    reach = SBS.LABEL_HALO_PX + 2
    assert dark[0] >= light[0] - reach
    assert dark[1] <= light[1] + reach, (
        f"the dark pass ends {dark[1] - light[1]} px after the caption: a doubled tail")

    # And the light pass is the plain one-stroke caption, end for end.
    plain = np.full_like(canvas, GREY)
    cv2.putText(plain, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, scale, SBS._FG, 1,
                cv2.LINE_AA)
    ref = _extent((plain > 200).all(axis=2))
    assert abs(light[0] - ref[0]) <= 1 and abs(light[1] - ref[1]) <= 1


def test_the_old_two_stroke_caption_did_draw_a_tail_under_this_opencv():
    """The teeth: the replaced drawing fails the same measurement here.

    If a future OpenCV advances glyphs independently of the stroke again, this
    skips rather than failing, because the defect it documents is gone either way.
    """
    text = "the twin sees only the recorded pressure targets"
    canvas = np.full((80, 1000, 3), GREY, dtype=np.uint8)
    cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, SBS._FG, 1,
                cv2.LINE_AA)
    light = _extent((canvas > 200).all(axis=2))
    dark = _extent((canvas < 40).all(axis=2))
    if dark[1] <= light[1] + SBS.LABEL_HALO_PX + 2:
        pytest.skip(f"OpenCV {cv2.__version__} advances glyphs independently of stroke")
    assert dark[1] - light[1] > 5
