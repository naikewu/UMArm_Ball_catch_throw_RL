r"""Fit the flow net by multiple shooting, on the GPU, with autograd.

**This is the only module in ``digital_twin`` allowed to import torch.**
``actuator_model`` is stepped by ``sim_core`` once per 1 ms quantum on a bench
laptop with no CUDA, so the trained model and the stepped model are two
implementations of one function, and the single thing that keeps them the same
function is ``test_train_actuator_net.py``'s equality test: the numpy forward and
the torch forward evaluated on the same weights must agree to float64 tolerance.
Nothing else checks it, and nothing else can -- a divergence between them shows
up as a twin that quietly disagrees with the checkpoint it loaded.

THE OBJECTIVE IS THE REFERENCE'S, UNCHANGED IN STRUCTURE (``reference/
actuator_net.md`` sections 2.6 and 2.9).  Multiple shooting on windows, each
window re-anchored at its own recorded ``p0``, the firmware's regulator inside
the loop, and

    loss = mean( ((p_sim - p_rec) / P_SCALE_PA)^2 )

over ticks 1..K-1.  Re-anchoring is what keeps a long recording from becoming one
error-accumulating rollout, and putting the regulator inside the loop is what
makes the fitted flow a property of the valve rather than of the trajectory: a
net fitted against recorded valve behaviour reproduces that recording and nothing
else.

WHAT IS DONE DIFFERENTLY, AND WHY.  The reference hand-wrote its backward pass
through the rollout in numpy and gradient-checked it, because it had no
alternative.  Here the same rollout is written once, forward, in torch, and the
adjoint is autograd's.  Three things follow, and only the first is a convenience:

* the state Jacobian is exact by construction.  On this arm ``p`` enters the
  feature row twice -- as column 0 and, with the opposite sign, inside the
  commanded error of column 1 -- so a hand-written adjoint that carried only
  ``dy/dx0`` (correct for the reference, whose row had no error column) would
  drop a term of the same order and the opposite sign and still train;
* the rollout runs over every window at once on the 5090 instead of over a few
  hundred, in float64 where the pressure recursion needs it: the recursion is
  ``p <- p + (g*f - leak)*h`` with ``h`` near 1 ms and ``p`` of order 1e5 Pa, so a
  float32 accumulation loses about 6 Pa per substep against a leak signal of
  about 0.3 Pa per substep;
* ``chunk`` accumulates the gradient over slices of the window set, so the
  full-batch Adam step of the reference is preserved while the tape of any one
  slice fits in VRAM.

WHAT THE FIRMWARE REGULATOR IS ON THIS BUS.  Not a three-way valve state -- that
is not observable here (CONTRACT departure 1) -- but the commanded error itself,
latched once per sync edge from the pressure the *sensor* reported and held while
the plant integrates.  That order is CONTRACT section 4's node pass, and it is
load-bearing rather than pedantic: recomputing the error every substep lets the
modelled valve shut the instant it crosses its dead band, removing the overshoot
that turns the band into an effective hysteresis, and with it every closed decay
the leak fit lives on.

WHAT A GREEN RUN OF THIS FILE DOES NOT SHOW.  It does not show that the twin is
right.  It shows that the pipeline recovers a plant it was given, that the two
forward passes agree, and that the held-out episodes were not in the training
set.  The tendon geometry may still be the placeholder (see
``dataset.TendonKinematics.auto``), the excitation may not cover the bent-arm
``l`` envelope, and the absolute scale of ``fill_gain``/``vent_gain`` is
degenerate with the shared net's output scale -- only between-node ratios are
identifiable, so no acceptance test may be written on an absolute gain.

Run::

    python -m digital_twin.train_actuator_net --session data/session_20260910_120000 \
        --out digital_twin/checkpoints/canarm_flow.npz --epochs 200 --device cuda
    python -m digital_twin.train_actuator_net --make-synthetic /tmp/s --epochs 40
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

try:
    from . import actuator_model as am
    from . import dataset as ds
except ImportError:  # pragma: no cover - depends on how the caller was started
    _WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _WS_ROOT not in sys.path:
        sys.path.insert(0, _WS_ROOT)
    from digital_twin import actuator_model as am  # type: ignore[no-redef]
    from digital_twin import dataset as ds  # type: ignore[no-redef]

__all__ = [
    "NET_KEYS", "TorchModel", "to_torch", "net_forward", "gain",
    "norm_x", "net_flow_pa_s_batch", "Batch", "make_batch", "shoot",
    "train_shooting", "train_warm_start", "build_warmstart_dataset",
    "evaluate", "fit_leaks_for", "fresh_model", "write_checkpoint", "main",
]

NET_KEYS = am.FlowNet.PARAM_KEYS          # ("W1","b1","W2","b2","W3","b3")

#: Integration substep, s.  1 ms, which is ``digital_twin.TIMESTEP_S``: the
#: rollout that the fit is scored against is the one ``sim_core`` will run, and
#: fitting at a coarser step fits the discretisation as well as the plant.  At
#: the CAN arm's 150 Hz sync rate this is 7 substeps per cycle -- the reference's
#: 5 ms would have given 1, i.e. no sub-cycle resolution at all, which is
#: ``reference/actuator_net.md`` porting note 3.
DEFAULT_SUBSTEP_S = 0.001

#: Windows per gradient-accumulation slice.  The tape holds, per substep,
#: ``(B,5)`` inputs and two ``(B,64)`` activations in float64; at 300 cycles x 7
#: substeps that is about 21 MB per 100 windows, so 512 keeps a slice near 100 MB
#: and leaves the 5090's 32 GB for the graph autograd builds around it.  It
#: changes nothing about the optimisation -- the step is still full batch.
DEFAULT_CHUNK = 512


# ---------------------------------------------------------------------------
# The torch mirror of actuator_model's forward
# ---------------------------------------------------------------------------
@dataclass
class TorchModel:
    """The fitted parameters as leaf tensors, plus the fixed per-node facts.

    Separate from :class:`actuator_model.ActuatorModel` rather than wrapping it,
    because the two exist for opposite reasons: that one is stepped a million
    times and must not allocate, this one is differentiated and must keep a tape.
    :func:`to_torch` and :func:`write_back` are the only two places they meet, and
    both copy rather than share so a half-finished epoch cannot leave a partially
    updated numpy model behind.
    """

    net: dict                  # NET_KEYS -> leaf tensor, requires_grad
    fill_gain: torch.Tensor    # (24,) leaf
    vent_gain: torch.Tensor    # (24,) leaf
    leak_pa_s: torch.Tensor    # (24,) fixed -- fitted by least squares, not here
    blend_width_pa: torch.Tensor   # (2,) fixed per population, never fitted
    is_tle: torch.Tensor       # (24,) float64 0/1, from the recorded variant byte
    device: torch.device
    dtype: torch.dtype

    def net_params(self) -> list:
        return [self.net[k] for k in NET_KEYS]

    def gain_params(self) -> list:
        return [self.fill_gain, self.vent_gain]

    def clip_gains_(self, gain_clip=am.GAIN_CLIP) -> None:
        """The reference's post-step clip, in place and outside autograd.

        A negative or exploding gain is a numerical accident and not a pneumatic
        circuit; the clip is applied after the step rather than as a penalty
        because the quantity is degenerate with the net's output scale and a
        penalty would just move the degeneracy.
        """
        with torch.no_grad():
            self.fill_gain.clamp_(gain_clip[0], gain_clip[1])
            self.vent_gain.clamp_(gain_clip[0], gain_clip[1])

    def write_back(self, model: "am.ActuatorModel") -> "am.ActuatorModel":
        """Copy every fitted array into a numpy :class:`ActuatorModel`, in place."""
        for k in NET_KEYS:
            getattr(model.net, k)[...] = self.net[k].detach().cpu().numpy()
        model.fill_gain[...] = self.fill_gain.detach().cpu().numpy()
        model.vent_gain[...] = self.vent_gain.detach().cpu().numpy()
        model.leak_pa_s[...] = self.leak_pa_s.detach().cpu().numpy()
        return model


def to_torch(model: "am.ActuatorModel", *, device="cpu",
             dtype=torch.float64, train_gains: bool = True) -> TorchModel:
    """Lift a numpy model onto a device.  Copies; the numpy model is untouched."""
    dev = torch.device(device)

    def leaf(a, req):
        t = torch.tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=dev)
        t.requires_grad_(req)
        return t

    return TorchModel(
        net={k: leaf(getattr(model.net, k), True) for k in NET_KEYS},
        fill_gain=leaf(model.fill_gain, train_gains),
        vent_gain=leaf(model.vent_gain, train_gains),
        leak_pa_s=leaf(model.leak_pa_s, False),
        blend_width_pa=leaf(model.blend_width_pa, False),
        is_tle=leaf(model.is_tle.astype(np.float64), False),
        device=dev, dtype=dtype)


def net_forward(net: dict, x: torch.Tensor) -> torch.Tensor:
    """``(B,5)`` -> ``(B,)``.  Line for line ``actuator_model.FlowNet.forward``.

    Written out rather than expressed as ``nn.Sequential`` so the correspondence
    with the numpy forward is readable at a glance: the equality test that keeps
    the two the same function is only as strong as a reader's ability to see that
    they are the same expression.
    """
    h1 = torch.tanh(x @ net["W1"] + net["b1"])
    h2 = torch.tanh(h1 @ net["W2"] + net["b2"])
    return (h2 @ net["W3"])[:, 0] + net["b3"][0]


def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    """The same sign-split logistic ``actuator_model._sigmoid`` uses.

    Split on the sign, not for elegance but so both branches take ``exp`` of a
    non-positive argument.  The blend argument is ``e / blend_width``, and the
    hard-switch limit test drives that to 2e14; ``exp`` of it overflows, and
    reproducing the numpy branch structure exactly is what makes the two paths
    agree bit for bit in the saturated tails rather than merely to tolerance.
    """
    u = torch.exp(-torch.abs(z))
    return torch.where(z >= 0, 1.0 / (1.0 + u), u / (1.0 + u))


def gain(tm: TorchModel, node_idx: torch.Tensor, e_pa: torch.Tensor) -> torch.Tensor:
    """CONTRACT section 2's logistic blend, verbatim.

    ``s = sigmoid(e / blend_width[pop])``, ``g = s*fill + (1-s)*vent``.  The blend
    replaces the reference's three-way branch on the valve state because the
    shooting loss back-propagates through that selection and a step at ``e = 0``
    puts a kink in the middle of the operating region -- and because a
    proportional TLE92464 valve has no discrete state to branch on.
    """
    pop = tm.is_tle[node_idx]                       # 1.0 TLE, 0.0 7 mm
    width = tm.blend_width_pa[0] * (1.0 - pop) + tm.blend_width_pa[1] * pop
    s = _sigmoid(e_pa / width)
    return s * tm.fill_gain[node_idx] + (1.0 - s) * tm.vent_gain[node_idx]


def norm_x(p_pa, e_pa, is_tle, l_m, ldot_m_s) -> torch.Tensor:
    """CONTRACT section 2's five-column feature row, in order.

    Takes ``e`` already formed rather than ``target``, because inside the shooting
    loop the error is the *latched* one -- the regulator acted on the pressure the
    sensor reported at the last sync edge, not on the true pressure at this
    substep -- and passing ``target`` would invite the loop to re-derive it from
    the true state.
    """
    return torch.stack((
        p_pa / am.P_SCALE_PA,
        e_pa / am.E_SCALE_PA,
        is_tle,
        l_m / am.L_SCALE_M,
        ldot_m_s / am.LDOT_SCALE_M_S,
    ), dim=1)


def net_flow_pa_s_batch(tm: TorchModel, node_idx, p_pa, target_pa, l_m,
                        ldot_m_s) -> torch.Tensor:
    """``dp/dt`` before the leak, Pa/s -- the torch twin of
    :meth:`actuator_model.ActuatorModel.net_flow_pa_s_batch`.

    This is the function the equality test compares.  It takes ``target`` rather
    than ``e`` and forms the error itself, exactly as the numpy path does, so the
    two are compared over the whole feature construction and not merely over the
    MLP.
    """
    e = target_pa - p_pa
    x = norm_x(p_pa, e, tm.is_tle[node_idx], l_m, ldot_m_s)
    y = net_forward(tm.net, x)
    return gain(tm, node_idx, e) * y * am.DP_SCALE_PA_S


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    """One shooting problem: every window as a row of tensors on one device."""

    node_idx: torch.Tensor      # (B,) long
    is_tle: torch.Tensor        # (B,) dtype 0/1
    p_rec: torch.Tensor         # (B,K) Pa
    target_pa: torch.Tensor     # (B,K) Pa
    l: torch.Tensor             # (B,K) m
    ldot: torch.Tensor          # (B,K) m/s
    h: torch.Tensor             # (B,K-1) s, per-window substep length
    nsub: np.ndarray            # (K-1,) int, substeps per tick, batch max
    zero_counts: torch.Tensor   # (B,) each board's own calibration
    counts_per_psi: torch.Tensor

    def __len__(self) -> int:
        return int(self.node_idx.shape[0])

    @property
    def n_ticks(self) -> int:
        return int(self.p_rec.shape[1])

    def slice(self, a: int, b: int) -> "Batch":
        return Batch(node_idx=self.node_idx[a:b], is_tle=self.is_tle[a:b],
                     p_rec=self.p_rec[a:b], target_pa=self.target_pa[a:b],
                     l=self.l[a:b], ldot=self.ldot[a:b], h=self.h[a:b],
                     nsub=self.nsub, zero_counts=self.zero_counts[a:b],
                     counts_per_psi=self.counts_per_psi[a:b])


def make_batch(windows: "ds.WindowSet", *, device="cpu", dtype=torch.float64,
               substep_s: float = DEFAULT_SUBSTEP_S) -> Batch:
    """Move a :class:`dataset.WindowSet` onto a device and pick the substepping.

    ``nsub`` is the batch maximum per tick and ``h = dt / nsub`` is per window, so
    every window lands exactly on its own recorded sync edges while the loop stays
    rectangular.  The recorded timestamps are the step lengths and are not
    resampled: the collector's jitter -- 6.666 ms median against a 25.069 ms worst
    interval on the legacy hour run -- is in the data on purpose, and a rollout
    stepped on a nominal grid drifts against the recording it is scored on.
    """
    dev = torch.device(device)

    def T(a, dt=None):
        return torch.tensor(np.ascontiguousarray(a), dtype=dt or dtype, device=dev)

    dt = np.diff(windows.t, axis=1)
    if not np.all(dt > 0.0):
        raise ValueError(
            "window timestamps must be strictly increasing; a non-positive dt "
            "means two cycles share a sync edge or the episode split let a "
            "window straddle a gap")
    nsub = np.maximum(1, np.rint(dt / float(substep_s)).astype(np.int64)).max(axis=0)
    h = dt / nsub[None, :]
    return Batch(
        node_idx=torch.tensor(windows.node_idx.astype(np.int64), device=dev),
        is_tle=T(windows.is_tle.astype(np.float64)),
        p_rec=T(windows.p_rec), target_pa=T(windows.target_pa),
        l=T(windows.l), ldot=T(windows.ldot), h=T(h), nsub=nsub,
        zero_counts=T(windows.zero_counts), counts_per_psi=T(windows.counts_per_psi))


# ---------------------------------------------------------------------------
# The shooting rollout
# ---------------------------------------------------------------------------
def _quantise(p_pa, zero_counts, counts_per_psi):
    """True pressure -> the pressure the board would report, Pa.

    The simulated ADC, with each board's own calibration and the wire's own 12-bit
    saturation.  ``torch.round`` has a zero derivative everywhere it is
    differentiable, which is exactly the reference's treatment of the
    quantisation as piecewise constant in the parameters -- the resulting gradient
    is the true one almost everywhere.  It is detached anyway, so that the
    treatment is a stated choice rather than a property of an operator's backward
    that a torch release could change.
    """
    with torch.no_grad():
        c = torch.round(zero_counts + p_pa / ds.PA_PER_PSI * counts_per_psi)
        c = torch.clamp(c, 0.0, float(ds.ADC_MAX))
        return (c - zero_counts) / counts_per_psi * ds.PA_PER_PSI


def shoot(tm: TorchModel, batch: Batch, *, adc_in_loop: bool = True,
          return_traj: bool = False):
    """Roll every window forward and return ``(loss, p_sim)``.

    ``loss`` is the mean squared pressure residual normalised by
    :data:`actuator_model.P_SCALE_PA`, over ticks 1..K-1 -- tick 0 is the anchor
    and contributes nothing, which is what makes this multiple shooting rather
    than a single long rollout.

    ``adc_in_loop`` puts the simulated sensor between the plant and the regulator.
    Leave it on: it is what the boards do, and it is why a modelled valve
    overshoots its dead band by a control period's worth of fill.  Turning it off
    makes the commanded error a differentiable function of the state, which is the
    configuration the finite-difference gradient test uses, because with the ADC
    in the loop the error is piecewise constant in the pressure and a difference
    quotient that steps across a count boundary measures the count and not the
    gradient.
    """
    b = len(batch)
    kn = batch.n_ticks
    p = batch.p_rec[:, 0]
    leak = tm.leak_pa_s[batch.node_idx]
    tle = tm.is_tle[batch.node_idx]
    p_rep = batch.p_rec[:, 0]
    traj = [p] if return_traj else None
    sq = p.new_zeros(())

    for k in range(kn - 1):
        # Node pass, CONTRACT section 4 order: the regulator acts on the pressure
        # the sensor reported at this sync edge, and holds it for the cycle.
        e = batch.target_pa[:, k] - (p_rep if adc_in_loop else p)
        lk = batch.l[:, k]
        ldk = batch.ldot[:, k]
        hk = batch.h[:, k]
        g = gain(tm, batch.node_idx, e)
        for _ in range(int(batch.nsub[k])):
            if not adc_in_loop:
                e = batch.target_pa[:, k] - p
                g = gain(tm, batch.node_idx, e)
            x = norm_x(p, e, tle, lk, ldk)
            y = net_forward(tm.net, x)
            p = torch.clamp(p + (g * y * am.DP_SCALE_PA_S - leak) * hk, min=0.0)
        if adc_in_loop:
            p_rep = _quantise(p, batch.zero_counts, batch.counts_per_psi)
        r = (p - batch.p_rec[:, k + 1]) / am.P_SCALE_PA
        sq = sq + (r * r).sum()
        if return_traj:
            traj.append(p)

    loss = sq / float(b * (kn - 1))
    return (loss, torch.stack(traj, dim=1)) if return_traj else (loss, None)


def _accumulate(tm: TorchModel, batch: Batch, chunk: int) -> float:
    """One full-batch gradient, assembled from slices that fit in VRAM.

    Each slice is weighted by its share of the windows, so the accumulated
    gradient is exactly the full-batch one and the optimiser step is the
    reference's -- ``chunk`` buys memory, not a different optimisation problem.
    """
    n = len(batch)
    total = 0.0
    for a in range(0, n, chunk):
        bcut = batch.slice(a, min(a + chunk, n))
        w = len(bcut) / n
        loss, _ = shoot(tm, bcut)
        (loss * w).backward()
        total += float(loss.detach()) * w
    return total


def train_shooting(tm: TorchModel, windows: "ds.WindowSet", *, epochs: int = 120,
                   lr: float = 1e-3, gains_lr: float = 1e-2,
                   substep_s: float = DEFAULT_SUBSTEP_S,
                   train_gains: bool = True, chunk: int = DEFAULT_CHUNK,
                   gain_clip=am.GAIN_CLIP, log_every: int = 10,
                   log=print) -> list:
    """The primary objective.  Returns the per-epoch loss list.

    Two Adams, the reference's split: the net at ``lr`` and the gains at
    ``gains_lr``, ten times larger because a gain is one number per board against
    a net shared by all twenty-four and would otherwise never move within the
    epoch budget.  The leak stays fixed -- it was fitted by least squares on
    closed holds and the shooting loss has no gradient signal that separates it
    from flow at 150 Hz, where one cycle's leak is about 2 Pa against a
    quantisation step of 113 Pa.

    No early stopping, no schedule, no regularisation, matching the reference: the
    "stop if worse" decision belongs one level up, where a held-out metric exists
    to make it against.
    """
    batch = make_batch(windows, device=tm.device, dtype=tm.dtype,
                       substep_s=substep_s)
    opt_net = torch.optim.Adam(tm.net_params(), lr=lr)
    opts = [opt_net]
    if train_gains and tm.fill_gain.requires_grad:
        opts.append(torch.optim.Adam(tm.gain_params(), lr=gains_lr))

    losses = []
    for ep in range(int(epochs)):
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = _accumulate(tm, batch, chunk)
        for o in opts:
            o.step()
        tm.clip_gains_(gain_clip)
        losses.append(loss)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            log(f"  shoot epoch {ep:4d}/{epochs}  loss {loss:.6e}  "
                f"rms {np.sqrt(loss) * am.P_SCALE_PA:8.1f} Pa")
    return losses


# ---------------------------------------------------------------------------
# Warm start -- single-interval dp/dt regression
# ---------------------------------------------------------------------------
def build_warmstart_dataset(windows: "ds.WindowSet", leak_pa_s, *,
                            blend_width_pa=am.DEFAULT_BLEND_WIDTH_PA,
                            sigma_pa: float = ds.LEAK_SIGMA_PA,
                            leak_max_pa_s: float = am.LEAK_CLIP_PA_S[1]):
    """Intervals the net can be regressed on directly, with their targets.

    Returns ``(node_idx, p_mid, e_mid, l_mid, ldot_mid, y_target)``, where
    ``y_target`` is what ``f_theta`` must return for ``gain*f - leak`` to
    reproduce the measured slope.  It is the reference's warm start with the valve
    one-hot removed and the sign rules rewritten on the commanded error, because
    this bus has no valve-state bits to filter on.

    THE FILTER, and what each rule throws away:

    * a board being asked to fill (``e`` above its band at both ends of the
      interval) may not have lost pressure beyond the noise floor;
    * a board being asked to vent may not have gained it;
    * a board inside its band -- valve nominally shut -- may not have moved more
      than a leak's worth plus noise.

    None of the three is a physical law of the plant; each is a statement about
    what a *clean* interval looks like, and an interval that breaks one is far
    more likely to be a dropped reply or a blip than a discovery.  The filter
    shapes only this dataset: the shooting loss sees every window.

    The midpoint convention is the reference's -- ``p_mid`` from the 3-smoothed
    trace, the slope from the raw one -- so the regression input sits at the
    interval centre where the finite-difference slope is second-order accurate.
    """
    t = windows.t
    dt = np.diff(t, axis=1)
    p = windows.p_rec
    ps = ds.smooth3(p, axis=1)
    p_mid = 0.5 * (ps[:, :-1] + ps[:, 1:])
    dpdt = np.diff(p, axis=1) / dt
    dp = np.diff(p, axis=1)
    l_mid = 0.5 * (windows.l[:, :-1] + windows.l[:, 1:])
    ldot_mid = 0.5 * (windows.ldot[:, :-1] + windows.ldot[:, 1:])
    e0 = windows.target_pa[:, :-1] - ps[:, :-1]
    e1 = windows.target_pa[:, 1:] - ps[:, 1:]
    e_mid = 0.5 * (e0 + e1)

    band = np.where(windows.is_tle[:, None], blend_width_pa[1], blend_width_pa[0])
    lim = 3.0 * sigma_pa
    filling = (e0 > band) & (e1 > band)
    venting = (e0 < -band) & (e1 < -band)
    holding = (np.abs(e0) <= band) & (np.abs(e1) <= band)
    ok = ((filling & (dp >= -lim))
          | (venting & (dp <= lim))
          | (holding & (np.abs(dp) <= lim + leak_max_pa_s * dt)))
    if not ok.any():
        raise ValueError(
            "the interval filter rejected every interval; either the recording "
            "is all transitions across the regulation band or sigma_pa is set "
            "far below this arm's quantisation step (113-123 Pa per count)")

    node = np.repeat(windows.node_idx[:, None], dt.shape[1], axis=1)[ok]
    leak = np.asarray(leak_pa_s, dtype=np.float64)[node]
    return (node, p_mid[ok], e_mid[ok], l_mid[ok], ldot_mid[ok],
            (dpdt[ok] + leak))


def train_warm_start(tm: TorchModel, windows: "ds.WindowSet", *, epochs: int = 300,
                     lr: float = 3e-3, chunk: int = 200_000,
                     log_every: int = 100, log=print) -> list:
    """Regress the net on single-interval slopes, gains held fixed.

    A warm start and not a fit.  It cannot separate a per-node gain from the
    shared net's output scale -- that degeneracy is exactly what the shooting
    stage resolves through the recurrence -- so the gains are frozen here and
    only the net moves.  Its value is that it puts the net in the right decade
    before the first BPTT epoch, and the reference measured its loss falling by
    three orders of magnitude in 300 full-batch Adam steps.

    Returns the per-epoch loss list, which is a *different* loss from
    :func:`train_shooting`'s and must not be compared with it: this one is a
    normalised flow residual, that one a normalised pressure residual.
    """
    node, p_mid, e_mid, l_mid, ldot_mid, dp_target = build_warmstart_dataset(
        windows, tm.leak_pa_s.detach().cpu().numpy(),
        blend_width_pa=tm.blend_width_pa.detach().cpu().numpy())

    dev, dt_ = tm.device, tm.dtype
    idx = torch.tensor(node.astype(np.int64), device=dev)
    P = torch.tensor(p_mid, dtype=dt_, device=dev)
    E = torch.tensor(e_mid, dtype=dt_, device=dev)
    L = torch.tensor(l_mid, dtype=dt_, device=dev)
    LD = torch.tensor(ldot_mid, dtype=dt_, device=dev)
    DP = torch.tensor(dp_target, dtype=dt_, device=dev)
    tle = tm.is_tle[idx]

    opt = torch.optim.Adam(tm.net_params(), lr=lr)
    n = int(idx.shape[0])
    losses = []
    for ep in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for a in range(0, n, chunk):
            s = slice(a, min(a + chunk, n))
            w = (s.stop - s.start) / n
            g = gain(tm, idx[s], E[s])
            y = net_forward(tm.net, norm_x(P[s], E[s], tle[s], L[s], LD[s]))
            r = (g * y * am.DP_SCALE_PA_S - DP[s]) / am.DP_SCALE_PA_S
            loss = (r * r).mean()
            (loss * w).backward()
            total += float(loss.detach()) * w
        opt.step()
        losses.append(total)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            log(f"  warm  epoch {ep:4d}/{epochs}  loss {total:.6e}")
    return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(tm: TorchModel, windows: "ds.WindowSet", *,
             substep_s: float = DEFAULT_SUBSTEP_S,
             chunk: int = DEFAULT_CHUNK) -> dict:
    """Open-loop rollout RMS on a window set, overall and per node.

    Reported in Pa and psi both, because the operator's limits are in psi and the
    fit's residuals are in Pa, and quoting only one of them has cost a
    conversation every time.  Per-node as well as overall, because a single
    aggregate over two valve populations describes neither -- and because one bad
    board (``0x110`` leaks from its supply side) will otherwise be averaged into
    an acceptable-looking total.
    """
    batch = make_batch(windows, device=tm.device, dtype=tm.dtype,
                       substep_s=substep_s)
    n = len(batch)
    sq = np.zeros(am.N_ACT)
    cnt = np.zeros(am.N_ACT)
    tot_sq = 0.0
    with torch.no_grad():
        for a in range(0, n, chunk):
            bc = batch.slice(a, min(a + chunk, n))
            _, traj = shoot(tm, bc, return_traj=True)
            err = (traj[:, 1:] - bc.p_rec[:, 1:]).cpu().numpy()
            nd = bc.node_idx.cpu().numpy()
            np.add.at(sq, nd, (err ** 2).sum(axis=1))
            np.add.at(cnt, nd, err.shape[1])
            tot_sq += float((err ** 2).sum())
    rms = float(np.sqrt(tot_sq / max(1.0, cnt.sum())))
    per = {int(j): float(np.sqrt(sq[j] / cnt[j])) for j in range(am.N_ACT)
           if cnt[j] > 0}
    return {"rms_pa": rms, "rms_psi": rms / ds.PA_PER_PSI,
            "per_node_rms_pa": per, "n_windows": n, "n_ticks": batch.n_ticks}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def fresh_model(rec: "ds.Recording", *, seed: int = 20260910) -> "am.ActuatorModel":
    """An unfitted model whose population mask comes from the recording.

    From the variant byte the boards answered, never from the id range: that is
    the one fact about this arm the id block is known to get wrong.
    """
    return am.ActuatorModel.fresh(
        is_tle=am.is_tle_from_variants(rec.board_type), seed=seed)


def fit_leaks_for(rec: "ds.Recording", model: "am.ActuatorModel", *,
                  min_duration_s: float = 2.0, log=print):
    """Fit the per-node leak on the recording's closed holds, in place.

    Fitted before the net and held fixed through it, which is the reference's
    order and is structural: the seam computes ``dp/dt = gain*f_theta - leak``, so
    a leak that moves while the net trains is a leak the net can absorb, and a net
    that has absorbed a leak self-inflates every idle board.
    """
    leak, info = ds.fit_leaks(rec, blend_width_pa=model.blend_width_pa,
                              initial_pa_s=model.leak_pa_s,
                              min_duration_s=min_duration_s)
    model.leak_pa_s[...] = leak
    fitted = [j for j in info if info[j]["fitted_pa_s"] is not None]
    log(f"  leak: {len(fitted)}/{am.N_ACT} boards fitted, "
        f"median {np.median(leak[fitted]) if fitted else float('nan'):.1f} Pa/s, "
        f"{am.N_ACT - len(fitted)} kept their incoming value")
    return leak, info


def write_checkpoint(path, model: "am.ActuatorModel", meta_extra: dict) -> str:
    """Save through :meth:`ActuatorModel.save`, so ``load`` is what defines the format.

    Deliberately not a hand-written ``np.savez`` here.  The checkpoint's only
    consumer is ``ActuatorModel.load``, whose unit guard raises unless every
    normalisation constant in the meta matches the module; writing the file
    through the same class that reads it is what makes that guard a check on the
    training run rather than a check on this function's memory of the format.
    """
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    model.save(path, meta_extra)
    return path if path.endswith(".npz") else path + ".npz"


def run(session_dir, out_path, *, epochs: int = 120, warm_epochs: int = 300,
        device: str = "auto", holdout_frac: float = 0.25,
        window_cycles: int = 300, substep_s: float = DEFAULT_SUBSTEP_S,
        lr: float = 1e-3, gains_lr: float = 1e-2, chunk: int = DEFAULT_CHUNK,
        seed: int = 20260910, dtype: str = "float64", warm_start: bool = True,
        train_gains: bool = True, tendon=None, log=print) -> dict:
    """Leak fit -> optional warm start -> shooting -> held-out evaluation -> checkpoint.

    Returns the metrics dictionary that is also written into the checkpoint's
    ``meta``, so a checkpoint carries the evidence for its own quality rather than
    leaving it in a terminal that has since scrolled.
    """
    t0 = time.perf_counter()
    dev = _resolve_device(device)
    tdtype = {"float64": torch.float64, "float32": torch.float32}[dtype]

    log(f"reading {session_dir}")
    if not isinstance(session_dir, (str, bytes, os.PathLike)):
        session_dir = list(session_dir)
    if tendon is None:
        tendon = ds.TendonKinematics.auto()
    placeholder = bool(getattr(tendon, "is_placeholder", False))
    if placeholder:
        log("  WARNING: tendon geometry is the moment-arm placeholder, not the "
            "twin's MJCF. l and ldot have the right units and the wrong "
            "geometry; this fit is a pipeline check, not a model of the arm.")
    train, hold, rec, _, _ = ds.build_windows(
        session_dir, window_cycles=window_cycles, holdout_frac=holdout_frac,
        seed=seed, tendon=tendon, log=log)
    log(f"  {rec.n_cycles} cycles, {rec.episode_ids().size} episodes, "
        f"{len(train)} train windows, {len(hold)} holdout windows, "
        f"{train.n_ticks} cycles each")
    log(f"  populations: {int(rec.is_tle.sum())} TLE, "
        f"{int((~rec.is_tle).sum())} 7 mm")

    model = fresh_model(rec, seed=seed)
    _, leak_info = fit_leaks_for(rec, model, log=log)

    tm = to_torch(model, device=dev, dtype=tdtype, train_gains=train_gains)
    base = evaluate(tm, hold, substep_s=substep_s, chunk=chunk)
    log(f"  baseline holdout RMS {base['rms_pa']:.1f} Pa "
        f"({base['rms_psi']:.3f} psi)")

    warm = []
    if warm_start and warm_epochs > 0:
        warm = train_warm_start(tm, train, epochs=warm_epochs, log=log)
    shoot_losses = train_shooting(tm, train, epochs=epochs, lr=lr,
                                  gains_lr=gains_lr, substep_s=substep_s,
                                  train_gains=train_gains, chunk=chunk, log=log)

    final = evaluate(tm, hold, substep_s=substep_s, chunk=chunk)
    train_metrics = evaluate(tm, train, substep_s=substep_s, chunk=chunk)
    log(f"  final    holdout RMS {final['rms_pa']:.1f} Pa "
        f"({final['rms_psi']:.3f} psi); train RMS {train_metrics['rms_pa']:.1f} Pa")

    tm.write_back(model)
    model.clip_parameters()
    metrics = {
        "kind": "canarm flow net, multiple shooting on windows",
        "session": ([os.path.abspath(str(d)) for d in session_dir]
                    if isinstance(session_dir, list)
                    else os.path.abspath(str(session_dir))),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed, "device": str(dev), "dtype": dtype,
        "epochs": epochs, "warm_epochs": warm_epochs if warm_start else 0,
        "window_cycles": int(train.n_ticks), "substep_s": substep_s,
        "holdout_frac": holdout_frac,
        "train_windows": len(train), "holdout_windows": len(hold),
        "train_episodes": sorted(set(int(e) for e in train.episode)),
        "holdout_episodes": sorted(set(int(e) for e in hold.episode)),
        "tendon_geometry": "moment-arm placeholder" if placeholder else "twin MJCF",
        "warm_loss_first_last": [warm[0], warm[-1]] if warm else None,
        "shoot_loss_first_last": [shoot_losses[0], shoot_losses[-1]],
        "baseline_holdout_rms_pa": base["rms_pa"],
        "holdout_rms_pa": final["rms_pa"], "holdout_rms_psi": final["rms_psi"],
        "train_rms_pa": train_metrics["rms_pa"],
        "holdout_per_node_rms_pa": final["per_node_rms_pa"],
        "leak_fitted_boards": sorted(j for j in leak_info
                                     if leak_info[j]["fitted_pa_s"] is not None),
        "train_seconds": time.perf_counter() - t0,
    }
    written = write_checkpoint(out_path, model, metrics)
    log(f"wrote {written}")
    metrics["checkpoint"] = written
    return metrics


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fit digital_twin.actuator_model's flow net by multiple "
                    "shooting on a CAN-arm recording.")
    ap.add_argument("--session", action="append", default=None,
                    help="session_*/ directory written by collection/; repeat "
                         "to fit several at once (their windows are "
                         "concatenated, their clocks are not)")
    ap.add_argument("--out", default="digital_twin/checkpoints/canarm_flow.npz",
                    help="checkpoint path; ActuatorModel.load must accept it")
    ap.add_argument("--epochs", type=int, default=120,
                    help="shooting/BPTT epochs, full batch")
    ap.add_argument("--warm-epochs", type=int, default=300,
                    help="single-interval warm-start epochs; 0 disables it")
    ap.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of whole EPISODES held out; never samples")
    ap.add_argument("--window-cycles", type=int, default=300,
                    help="shooting horizon in 150 Hz cycles (300 = 2.0 s)")
    ap.add_argument("--substep-s", type=float, default=DEFAULT_SUBSTEP_S)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help="windows per gradient-accumulation slice (VRAM only)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gains-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--no-gains", action="store_true",
                    help="freeze fill/vent gains (one-rig data cannot identify them)")
    ap.add_argument("--make-synthetic", metavar="DIR",
                    help="generate a synthetic session in DIR and fit that; "
                         "for exercising the pipeline with no hardware")
    ap.add_argument("--synthetic-episodes", type=int, default=6)
    ap.add_argument("--synthetic-seconds", type=float, default=12.0)
    args = ap.parse_args(argv)

    session = args.session
    if args.make_synthetic:
        session = ds.write_synthetic_session(
            os.path.join(args.make_synthetic, "session_synthetic"),
            n_episodes=args.synthetic_episodes,
            episode_s=args.synthetic_seconds, seed=args.seed)
        print(f"generated synthetic session at {session}")
    if not session:
        ap.error("one of --session or --make-synthetic is required")

    run(session, args.out, epochs=args.epochs, warm_epochs=args.warm_epochs,
        device=args.device, holdout_frac=args.holdout,
        window_cycles=args.window_cycles, substep_s=args.substep_s,
        lr=args.lr, gains_lr=args.gains_lr, chunk=args.chunk, seed=args.seed,
        dtype=args.dtype, warm_start=args.warm_epochs > 0,
        train_gains=not args.no_gains)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
