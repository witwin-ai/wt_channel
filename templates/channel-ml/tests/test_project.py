"""The Project files fit together: Scene parameters, Studies and the Workflow."""
from pathlib import Path

import witwin.studio as ws
from witwin.studio.workflows import find_workflow

PROJECT = Path(__file__).resolve().parents[1]


def test_receiver_position_follows_the_parameters():
    scene = ws.open_project(PROJECT).scene()
    assert (scene.parameters["rx_x"], scene.parameters["rx_z"]) == (8.0, 0.0)
    scene.parameters.set("rx_x", "-3.5")   # in memory only; nothing is saved
    scene.parameters.set("rx_z", "7.25")
    assert scene.objects["rx"].transform.position == (-3.5, 1.5, 7.25)


def test_studies_load_and_vary_scene_parameters():
    project = ws.open_project(PROJECT)
    assert project.studies.problems() == []
    dataset = project.studies.get("link-gain")
    assert str(dataset.kind) == "dataset" and dataset.method.kind == "monte_carlo"
    assert set(dataset.variables) <= set(project.scene(dataset.scene).parameters)
    assert (PROJECT / dataset.export.path).is_file()
    train = project.studies.get("train-gain")
    assert str(train.kind) == "train" and (PROJECT / train.script).is_file()
    assert train.inputs["dataset"] == dataset.dataset.dir


def test_workflow_chains_the_studies():
    workflow = find_workflow(PROJECT, "channel-pipeline")
    steps = {step.id: step for step in workflow.steps}
    assert [step.id for step in workflow.steps] == ["synthesize", "train", "evaluate"]
    assert steps["synthesize"].study == "link-gain"
    assert steps["train"].study == "train-gain" and len(steps["train"].matrix["lr"]) == 3
    assert (PROJECT / steps["evaluate"].entry).is_file()
