"""Third-party code, kept verbatim so a diff against its origin stays readable.

``kinova_driver.py`` and ``utilities.py`` are copies of the lab's portable
Kinova package (``UMARM_Variable_Stiffness_Oct2025/kinova_arm_driver/``,
2026-05-26), which is itself distilled from Kinova's own BSD-3 examples.
``PROVENANCE.md`` records where each file came from and the one line that was
changed.  Nothing in this package should be edited to fix a repo-side problem:
put the fix in :mod:`UMArm_KINOVA.kinova_arm` instead, so the next re-vendoring
is a straight copy.
"""

from .kinova_driver import KinovaArm, pose_to_SE3, SE3_to_pose  # noqa: F401

__all__ = ["KinovaArm", "pose_to_SE3", "SE3_to_pose"]
