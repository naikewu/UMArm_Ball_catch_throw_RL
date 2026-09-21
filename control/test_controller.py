"""Controller invariants: measured wiring, constrained commands and learned parity."""
import json
import numpy as np
import pytest

from .controller import (CAP_PA, PA_PER_PSI, InverseDynamics, assert_pressures,
                         differential_to_pressure, make_controller,
                         pair_indices, project_pressures)


def test_pressure_projection_uses_measured_pairs_and_is_idempotent():
    rng=np.random.default_rng(131)
    raw=rng.normal(10,60,(600,24))*PA_PER_PSI
    safe=project_pressures(raw)
    for row in safe: assert_pressures(row)
    np.testing.assert_allclose(project_pressures(safe),safe,atol=1e-8)
    assert tuple(pair_indices()[0])==(7,3) # CAN's measured rotated upper platform.


@pytest.mark.parametrize("bad",[np.full(24,np.nan),np.ones(23),np.full(24,-1.),np.full(24,CAP_PA)])
def test_bad_pressure_refused(bad):
    with pytest.raises(ValueError): assert_pressures(bad)


def test_positive_difference_goes_to_measured_positive_board():
    for j,(a,b) in enumerate(pair_indices()):
        difference=np.zeros(12);difference[j]=8*PA_PER_PSI
        p=differential_to_pressure(difference)
        assert p[a]>p[b]
        assert_pressures(p)


def test_inverse_model_q_order_and_static_balance():
    inverse=InverseDynamics()
    q=np.deg2rad(np.linspace(-3,3,12))
    p,residual=inverse.allocate(q,np.zeros(12),np.zeros(12),np.ones(24)*6*PA_PER_PSI)
    np.testing.assert_allclose(inverse.arm.q(),q,atol=1e-12)
    assert residual<1e-6
    assert_pressures(p)


@pytest.mark.parametrize("name",["pid","ff_pid"])
def test_controllers_bound_integrator_and_commands(name):
    c=make_controller(name);q=np.zeros(12)
    c.reset(q,np.zeros(24))
    for _ in range(80):
        p=c.command(q,q,np.zeros(24),np.full(12,.2),q,q)
        assert_pressures(p)
    assert max(abs(c.integral))<=.3+1e-12
    c.reset(q,np.zeros(24));assert not c.integral.any()


def test_numpy_and_torch_prediction_are_same_model():
    import torch
    from .koopman import LiftedDynamics,NumpyPredictor
    torch.manual_seed(121)
    model=LiftedDynamics().double()
    with torch.no_grad():
        model.W.normal_(0,.001)
        model.B.normal_(0,.01)
        model.xmean.normal_(0,.1)
        model.xscale.uniform_(.1,2)
    rng=np.random.default_rng(314)
    x=rng.normal(size=(5,48));u=rng.uniform(0,CAP_PA,(5,24))
    predictor=NumpyPredictor(model)
    xt=torch.tensor(x);ut=torch.tensor(u)
    z=model.encode(xt)
    np.testing.assert_allclose(predictor.encode(x),z.detach().numpy(),atol=1e-12)
    expected=model.decode(model.step(z,ut)).detach().numpy()
    actual=predictor.decode(predictor.step(predictor.encode(x),u))
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_episode_split_is_disjoint_and_preserves_families():
    from .train import split_episodes
    episodes=[{"meta":{"family":f,"episode":j*100+i}} for j,f in enumerate(["chirp","prbs"]) for i in range(12)]
    splits=split_episodes(episodes)
    ids={name:{e["meta"]["episode"] for e in es} for name,es in splits.items()}
    assert len(ids["train"])==16 and len(ids["test"])==4 and len(ids["validation"])==4
    assert not ids["train"]&ids["validation"] and not ids["train"]&ids["test"] and not ids["test"]&ids["validation"]
    assert all({e["meta"]["family"] for e in es}=={"chirp","prbs"} for es in splits.values())


def test_checkpoint_schema_refuses_old_period(tmp_path):
    import torch
    from .koopman import LiftedDynamics,load_model,SCHEMA
    m=LiftedDynamics();path=tmp_path/"bad.pt"
    torch.save({"config":m.config(),"state_dict":m.state_dict(),"meta":{"schema":SCHEMA,"dt_s":1/160}},path)
    with pytest.raises(ValueError,match="period"):load_model(path)


def test_training_refuses_unlabeled_units(tmp_path):
    from .train import load_episodes
    np.savez(tmp_path/"episode_0000.npz",state=np.zeros((3,48)),action=np.zeros((2,24)),
             meta=json.dumps({"dt_s":1/150,"state_order":"q_deg,pressure_psi"}))
    with pytest.raises(ValueError,match="units"):load_episodes(tmp_path)


def test_mppi_reset_replays_sampling(tmp_path):
    import torch
    from .koopman import LiftedDynamics,KoopmanMPPIController,SCHEMA
    m=LiftedDynamics();path=tmp_path/"test.pt"
    meta={"schema":SCHEMA,"status":"complete","dt_s":1/150,"state_order":"q12,qdot12,p24_Pa_gauge",
          "pair_indices":pair_indices().tolist(),"variants":[2]*8+[0]*16}
    torch.save({"config":m.config(),"state_dict":m.state_dict(),"meta":meta},path)
    c=KoopmanMPPIController(checkpoint=path,horizon=4,samples=8)
    q=np.zeros(12);p=np.zeros(24)
    c.reset(q,p);a=c.command(q,q,p,q,q,q)
    c.command(q,q,p,q,q,q)
    c.reset(q,p);b=c.command(q,q,p,q,q,q)
    np.testing.assert_array_equal(a,b)


def test_zero_mppi_correction_blend_preserves_feedforward_prior(tmp_path):
    import torch
    from .koopman import LiftedDynamics, KoopmanMPPIController, SCHEMA
    model = LiftedDynamics()
    path = tmp_path / "test.pt"
    meta = {
        "schema": SCHEMA,
        "status": "complete",
        "dt_s": 1 / 150,
        "state_order": "q12,qdot12,p24_Pa_gauge",
        "pair_indices": pair_indices().tolist(),
        "variants": [2] * 8 + [0] * 16,
    }
    torch.save({"config": model.config(), "state_dict": model.state_dict(), "meta": meta}, path)
    feedforward = make_controller("ff_pid")
    mppi = KoopmanMPPIController(
        checkpoint=path, horizon=4, samples=8, correction_blend=0.0
    )
    q = np.zeros(12)
    qdot = np.zeros(12)
    pressure = np.full(24, 6 * PA_PER_PSI)
    q_ref = np.linspace(-.03, .03, 12)
    qd_ref = np.linspace(.10, -.10, 12)
    qdd_ref = np.zeros(12)
    feedforward.reset(q, pressure)
    mppi.reset(q, pressure)
    expected = feedforward.command(q, qdot, pressure, q_ref, qd_ref, qdd_ref)
    actual = mppi.command(
        q, qdot, pressure, q_ref, qd_ref, qdd_ref,
        future_q=np.repeat(q_ref[None, :], 4, axis=0),
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)
