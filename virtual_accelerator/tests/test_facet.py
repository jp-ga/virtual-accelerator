import os
import importlib.util

import pytest
from virtual_accelerator.tests._bmad_model_test_utils import (
    HAS_BMAD_DEPS,
    TEST_BEAM_PATH,
    assert_bmad_model_initialization,
    assert_bmad_model_track_beam_custom_path,
    assert_bmad_model_twiss_outputs,
)
from virtual_accelerator.models.facet2 import get_facet_bmad_model


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_FACET_LATTICE = bool(os.environ.get("FACET2_LATTICE"))


@pytest.mark.skipif(
    (not HAS_BMAD_DEPS) or (not HAS_FACET_LATTICE),
    reason="requires bmad optional dependencies and FACET2_LATTICE",
)
class TestFACET2Bmad:
    def test_initialization(self):
        assert_bmad_model_initialization(get_facet_bmad_model)

    def test_twiss(self):
        assert_bmad_model_twiss_outputs(get_facet_bmad_model)

    def test_track_beam_custom_path(self):
        assert_bmad_model_track_beam_custom_path(get_facet_bmad_model)

    def test_screen_variables(self):
        model = get_facet_bmad_model(
            track_beam=True, custom_beam_path=TEST_BEAM_PATH, end_element="PR10711"
        )
        # Check that screen variables are included in supported variables when tracking is enabled
        assert "OTRS:IN10:571:Image:ArrayData" in model.supported_variables
        assert "OTRS:IN10:711:Image:ArrayData" in model.supported_variables

        # test specific output from one of the screens to ensure it's properly set up
        output = model.get("OTRS:IN10:571:Image:ArrayData")
        assert output.shape == (1392, 1040)
