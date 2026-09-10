"""Real-system data collection for the CAN arm.

Four modules, in dependency order:

| module | role |
| --- | --- |
| `safety.py` | the operator's pressure envelope, and the only place it is enforced |
| `excitation.py` | the designed signals, each justified by what it makes identifiable |
| `recorder.py` | the 150 Hz JSONL session writer, on the documented schema |
| `campaign.py` | the hardware session: bus, mocap, camera, anomaly watch |

`campaign.py --dry-run` opens no port and checks the whole plan against the
envelope, which is how a change to `excitation.py` is verified before the arm is
asked to perform it.
"""

from .safety import IDLE_PSI, PAIR_SUM_MAX_PSI, SINGLE_MAX_PSI, PairEnvelope

__all__ = ["PairEnvelope", "IDLE_PSI", "PAIR_SUM_MAX_PSI", "SINGLE_MAX_PSI"]
