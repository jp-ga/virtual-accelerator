import importlib.util
import os
from pathlib import Path


TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "../bmad", "test_beam")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_BMAD_DEPS = has_module("pytao") and has_module("lume_bmad")


def assert_bmad_model_initialization(
    get_model,
    required_control_variable: str | None = None,
) -> None:
    model = get_model(custom_beam_path=TEST_BEAM_PATH)

    assert len(model.control_variables) > 0
    if required_control_variable is not None:
        assert required_control_variable in model.control_variables


def assert_bmad_model_twiss_outputs(get_model) -> None:
    model = get_model(custom_beam_path=TEST_BEAM_PATH)
    outputs = model.get(["a.beta", "b.beta", "name"])

    assert len(outputs["a.beta"]) == len(model.tao.lat_list("*", "ele.name"))
    assert len(outputs["b.beta"]) == len(model.tao.lat_list("*", "ele.name"))
    assert outputs["name"][0] == "BEGINNING"
    assert outputs["name"][-1] == "END"


def assert_bmad_model_track_beam_custom_path(get_model) -> None:
    # This test ensures shared track_beam setup works when custom_beam_path is given.
    model = get_model(track_beam=True, custom_beam_path=TEST_BEAM_PATH)
    assert model is not None
