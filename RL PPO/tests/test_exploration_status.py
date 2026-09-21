import json

import pytest

from teacher_rl import exploration_status as status


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(status,"source_hash",lambda:"source")
    monkeypatch.setattr(status,"file_hash",lambda path:"anchor")
    monkeypatch.setattr(status,"checkpoint_info",lambda path,*args:status.read_json(path))
    def write(name, data):
        (tmp_path/name).write_text(json.dumps(data),encoding="utf-8")
    write("run_config.json",dict(config=dict(stage="throw"),source_hash="source",
        anchor_sha256="anchor",budget_updates=2,episodes_per_update=5,eval_every=1))
    write("ppo_latest.pt",dict(update=2,best_update=0,accepted_update=None))
    write("ppo_candidate.pt",dict(update=0,best_update=0,accepted_update=None))
    for name in ("validation_0000.json","validation_0001.json","validation_0002.json",
            "update_0001.json","update_0002.json"):
        write(name,{})
    write("training_summary.json",dict(completed_updates=2,best_update=0,accepted_update=None))
    return tmp_path,write


def test_no_run_does_not_create_files(tmp_path):
    path = tmp_path/"absent"
    assert status.inspect_run(path)["state"]=="not_started"
    assert not path.exists()


def test_completed_zero_candidate_is_not_reported_as_trained(run):
    path,_ = run
    result = status.inspect_run(path)
    assert result["state"]=="completed" and not result["issues"]
    assert result["training_episodes"]==10
    assert result["checkpoints"]["latest"]["update"]==2
    assert "NOT a learned improvement" in result["warning"]


@pytest.mark.parametrize("name",["update_0001.json","validation_0002.json","ppo_candidate.pt","ppo_latest.pt"])
def test_missing_artifact_cannot_report_complete(run,name):
    path,_ = run
    (path/name).unlink()
    result = status.inspect_run(path)
    assert result["state"]=="inconsistent" and result["issues"]


def test_complete_updates_but_missing_summary(run):
    path,_ = run
    (path/"training_summary.json").unlink()
    assert status.inspect_run(path)["state"]=="updates_complete_summary_missing"


def test_partial_training_is_resumable(run):
    path,write = run
    (path/"training_summary.json").unlink()
    write("ppo_latest.pt",dict(update=1,best_update=0,accepted_update=None))
    assert status.inspect_run(path)["state"]=="resumable"


def test_interrupted_selection_detected(run):
    path,write = run
    write("ppo_candidate.pt",dict(update=2,best_update=2,accepted_update=None))
    assert status.inspect_run(path)["state"]=="inconsistent"


def test_missing_accepted_detected(run):
    path,write = run
    write("ppo_latest.pt",dict(update=2,best_update=0,accepted_update=2))
    assert status.inspect_run(path)["state"]=="inconsistent"


def test_source_mismatch_does_not_try_loading_models(run,monkeypatch):
    path,_ = run
    monkeypatch.setattr(status,"source_hash",lambda:"changed")
    monkeypatch.setattr(status,"checkpoint_info",lambda *args:pytest.fail("Must not load incompatible models"))
    assert status.inspect_run(path)["state"]=="source_or_anchor_mismatch"


def test_evaluation_tracks_selected_update_and_completion(run):
    path,_ = run
    evaluation = path/"evaluation_latest_40"
    evaluation.mkdir()
    (evaluation/"evaluation_contract.json").write_text('{"selected_update":2}',encoding="utf-8")
    assert status.inspect_run(path)["evaluations"]==[
        dict(directory="evaluation_latest_40",selected_update=2,complete=False)]
    (evaluation/"comparison.json").write_text('{}',encoding="utf-8")
    assert status.inspect_run(path)["evaluations"][0]["complete"]
