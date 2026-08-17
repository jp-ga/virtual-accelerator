import json

import pytest

from virtual_accelerator.tests.dependency_profiles import HAS_CHEETAH_DEPS

pytestmark = [
    pytest.mark.requires_cheetah,
]

if HAS_CHEETAH_DEPS:
    import torch
    from cheetah.accelerator import Drift, Quadrupole, Screen, Segment
    from cheetah.particles import ParticleBeam

    from virtual_accelerator.cheetah.factory import (
        CheetahModelSpec,
        apply_screen_geometry,
        build_cheetah_model,
        build_cheetah_model_from_json,
        load_profmon_config,
        resolve_device_mapping,
    )
else:
    pytest.skip("requires cheetah optional dependencies", allow_module_level=True)


NAME_MAP = {"Q1": "QUAD:IN10:511", "PR1": "OTRS:IN10:571"}

PROFMON_YAML = """PR1:
   name: PROF:IN10:571
   shape:
   - 128
   - 96
   pixel_size: 11.1732
"""

# Minimal stand-in for the LCLS elements table read by get_mad_control_mapping.
ELEMENTS_CSV = """Element,Control System Name
Q1,QUAD:IN10:511
PR1,OTRS:IN10:571
"""


@pytest.fixture
def segment():
    return Segment(
        [
            Quadrupole(
                name="Q1",
                length=torch.tensor(0.5),
                k1=torch.tensor(1.0),
            ),
            Drift(name="D1", length=torch.tensor(1.0)),
            Screen(
                name="PR1",
                resolution=(64, 48),
                pixel_size=torch.tensor((5e-6, 5e-6)),
                is_active=True,
            ),
        ],
        name="TEST",
    )


@pytest.fixture
def lattice_path(segment, tmp_path):
    path = tmp_path / "lattice.json"
    segment.to_lattice_json(str(path))
    return path


@pytest.fixture
def lattice_json(lattice_path):
    return lattice_path.read_text()


@pytest.fixture
def profmon_path(tmp_path):
    path = tmp_path / "test_profmon_info.yaml"
    path.write_text(PROFMON_YAML)
    return path


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "lcls_elements.csv"
    path.write_text(ELEMENTS_CSV)
    return path


class TestLatticeSources:
    def test_inline_json_spec_is_json_pure(self, lattice_json):
        spec = {
            "feature": "test model",
            "lattice": lattice_json,
            "name_map": NAME_MAP,
        }
        # A spec that survives json round-tripping is one a serialized model can
        # carry without pickling anything.
        json.dumps(spec)

        model = build_cheetah_model(spec, energy=1.25e8)

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables
        assert "OTRS:IN10:571:Image:ArrayData" in model.supported_variables

    def test_dict_and_dataclass_specs_agree(self, lattice_json):
        from_dict = build_cheetah_model(
            {"lattice": lattice_json, "name_map": NAME_MAP}, energy=1e8
        )
        from_dataclass = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json, name_map=NAME_MAP), energy=1e8
        )

        assert set(from_dict.supported_variables) == set(
            from_dataclass.supported_variables
        )

    def test_live_segment_is_used_directly(self, segment):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=segment, name_map=NAME_MAP), energy=1e8
        )

        assert model.simulator.segment is segment

    def test_lattice_path_is_loaded(self, lattice_path):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=str(lattice_path), name_map=NAME_MAP), energy=1e8
        )

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables

    def test_env_var_route(self, lattice_path, monkeypatch):
        monkeypatch.setenv("TEST_LATTICE_ROOT", str(lattice_path.parent))
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice_env_var="TEST_LATTICE_ROOT",
                lattice_relpath=lattice_path.name,
                database_relpath=None,
                name_map=NAME_MAP,
            ),
            energy=1e8,
        )

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables

    def test_spec_json_file(self, lattice_json, tmp_path):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(
            json.dumps({"lattice": lattice_json, "name_map": NAME_MAP})
        )

        model = build_cheetah_model_from_json(str(spec_path), energy=1e8)

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables

    def test_unset_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("TEST_LATTICE_ROOT", raising=False)

        with pytest.raises(ValueError, match="TEST_LATTICE_ROOT"):
            build_cheetah_model(
                CheetahModelSpec(
                    lattice_env_var="TEST_LATTICE_ROOT",
                    lattice_relpath="lattice.json",
                ),
                energy=1e8,
            )

    def test_no_lattice_raises(self):
        with pytest.raises(ValueError, match="no lattice"):
            build_cheetah_model(CheetahModelSpec(name_map=NAME_MAP), energy=1e8)

    def test_non_lattice_type_raises(self):
        with pytest.raises(TypeError, match="lattice"):
            build_cheetah_model(
                CheetahModelSpec(lattice=42, name_map=NAME_MAP), energy=1e8
            )


class TestDeviceMapping:
    def test_names_from_elements_table(self, database_path):
        mapping = resolve_device_mapping(
            CheetahModelSpec(database_path=str(database_path))
        )

        assert mapping == {"Q1": "QUAD:IN10:511", "PR1": "OTRS:IN10:571"}

    def test_missing_elements_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_device_mapping(
                CheetahModelSpec(database_path=str(tmp_path / "absent.csv"))
            )

    def test_precedence_table_then_name_map_then_profmon_then_screens(
        self, database_path, profmon_path
    ):
        # The table says OTRS, the profmon config says PROF: the profmon name is
        # what the FACET control system (and its archived data) actually uses.
        mapping = resolve_device_mapping(
            CheetahModelSpec(
                database_path=str(database_path),
                profmon_config_filename=str(profmon_path),
            )
        )
        assert mapping["PR1"] == "PROF:IN10:571"

        # name_map overrides the table but not the profmon screen name...
        mapping = resolve_device_mapping(
            CheetahModelSpec(
                database_path=str(database_path),
                name_map={"Q1": "QUAD:LI10:1", "PR1": "FROM:NAME:MAP"},
                profmon_config_filename=str(profmon_path),
            )
        )
        assert mapping["Q1"] == "QUAD:LI10:1"
        assert mapping["PR1"] == "PROF:IN10:571"

        # ...and `screens` overrides everything, as the documented escape hatch.
        mapping = resolve_device_mapping(
            CheetahModelSpec(
                database_path=str(database_path),
                name_map={"PR1": "FROM:NAME:MAP"},
                profmon_config_filename=str(profmon_path),
                screens={"PR1": "FROM:SCREENS:1"},
            )
        )
        assert mapping["PR1"] == "FROM:SCREENS:1"

    def test_empty_mapping_raises(self, lattice_json):
        with pytest.raises(ValueError, match="empty element"):
            build_cheetah_model(CheetahModelSpec(lattice=lattice_json), energy=1e8)

    def test_profmon_renames_screen_pvs(self, lattice_json, profmon_path):
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice=lattice_json,
                name_map=NAME_MAP,
                profmon_config_filename=str(profmon_path),
            ),
            energy=1e8,
        )

        assert "PROF:IN10:571:Image:ArrayData" in model.supported_variables
        assert "OTRS:IN10:571:Image:ArrayData" not in model.supported_variables

    def test_shipped_profmon_config_is_loadable(self):
        config = load_profmon_config("facet2_profmon_info.yaml")

        assert config["PR10571"]["name"] == "PROF:IN10:571"


class TestScreenGeometry:
    def test_lattice_geometry_is_kept_by_default(self, lattice_json, profmon_path):
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice=lattice_json,
                name_map=NAME_MAP,
                profmon_config_filename=str(profmon_path),
            ),
            energy=1e8,
        )

        # Only the name is taken from the profmon config: its shape axis order is
        # not consistent across tables, so geometry stays with the lattice.
        assert tuple(model.simulator.segment.PR1.resolution) == (64, 48)
        assert model.supported_variables["PROF:IN10:571:Image:ArrayData"].shape == (
            64,
            48,
        )

    def test_screen_geometry_opt_in(self, lattice_json, profmon_path):
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice=lattice_json,
                name_map=NAME_MAP,
                profmon_config_filename=str(profmon_path),
                screen_geometry=True,
            ),
            energy=1e8,
        )
        screen = model.simulator.segment.PR1

        assert tuple(screen.resolution) == (128, 96)
        # Config is in micrometers, Cheetah in meters.
        assert float(screen.pixel_size[0]) == pytest.approx(11.1732e-6)
        assert model.supported_variables["PROF:IN10:571:Image:ArrayData"].shape == (
            128,
            96,
        )
        # RESOLUTION reads back in micrometers, matching the config.
        assert float(model.get("PROF:IN10:571:RESOLUTION")) == pytest.approx(
            11.1732, rel=1e-6
        )
        assert tuple(model.get("PROF:IN10:571:Image:ArrayData").shape) == (128, 96)

    def test_screen_geometry_without_profmon_raises(self, lattice_json):
        with pytest.raises(ValueError, match="screen_geometry"):
            build_cheetah_model(
                CheetahModelSpec(
                    lattice=lattice_json,
                    name_map=NAME_MAP,
                    screen_geometry=True,
                ),
                energy=1e8,
            )

    def test_unconfigured_screen_keeps_lattice_geometry(self, segment):
        updated = apply_screen_geometry(segment, {"SOMETHING_ELSE": {"shape": [8, 8]}})

        assert updated == []
        assert tuple(segment.PR1.resolution) == (64, 48)


class TestIncomingBeam:
    def test_energy_builds_placeholder_beam(self, lattice_json):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json, name_map=NAME_MAP), energy=1.25e8
        )

        assert float(model.simulator.initial_beam_distribution.energy) == 1.25e8

    def test_supplied_beam_is_used(self, lattice_json):
        beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=64,
            energy=torch.tensor(1e6),
        )

        model = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json, name_map=NAME_MAP),
            initial_beam_distribution=beam,
        )

        # The simulator keeps its own copy, so compare contents rather than identity.
        initial_beam = model.simulator.initial_beam_distribution
        assert float(initial_beam.energy) == float(beam.energy)
        assert initial_beam.particles.shape == beam.particles.shape

    def test_no_beam_and_no_energy_raises(self, lattice_json):
        with pytest.raises(ValueError, match="incoming beam"):
            build_cheetah_model(
                CheetahModelSpec(lattice=lattice_json, name_map=NAME_MAP)
            )


class TestElementAttrMapping:
    def test_explicit_mapping_limits_variables(self, lattice_json):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json, name_map=NAME_MAP),
            energy=1e8,
            element_attr_mapping={"Quadrupole": {"BCTRL": "QuadrupoleBCTRLVariable"}},
        )

        quad_pvs = {
            name
            for name in model.supported_variables
            if name.startswith("QUAD:IN10:511")
        }
        assert quad_pvs == {"QUAD:IN10:511:BCTRL"}
