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
    )
else:
    pytest.skip("requires cheetah optional dependencies", allow_module_level=True)


PROFMON_YAML = """PR1:
   name: PROF:IN10:571
   shape:
   - 128
   - 96
   pixel_size: 11.1732
"""


@pytest.fixture
def segment():
    return Segment(
        [
            Quadrupole(
                name="Q1",
                length=torch.tensor(0.5),
                k1=torch.tensor(1.0),
                metadata={"alias": "QUAD:IN10:511"},
            ),
            Drift(name="D1", length=torch.tensor(1.0)),
            Screen(
                name="PR1",
                resolution=(64, 48),
                pixel_size=torch.tensor((5e-6, 5e-6)),
                is_active=True,
                metadata={"alias": "OTRS:IN10:571"},
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


class TestLatticeSources:
    def test_inline_json_spec_is_json_pure(self, lattice_json):
        spec = {
            "feature": "test model",
            "lattice": lattice_json,
        }
        # A spec that survives json round-tripping is one a serialized model can
        # carry without pickling anything.
        json.dumps(spec)

        model = build_cheetah_model(spec, energy=1.25e8)

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables
        assert "OTRS:IN10:571:Image:ArrayData" in model.supported_variables

    def test_dict_and_dataclass_specs_agree(self, lattice_json):
        from_dict = build_cheetah_model({"lattice": lattice_json}, energy=1e8)
        from_dataclass = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json), energy=1e8
        )

        assert set(from_dict.supported_variables) == set(
            from_dataclass.supported_variables
        )

    def test_spec_fields_as_keywords_agree(self, lattice_json):
        # A caller storing the recipe as flat kwargs splats it in directly.
        config = {"lattice": lattice_json}
        from_kwargs = build_cheetah_model(**config, energy=1e8)
        from_spec = build_cheetah_model(config, energy=1e8)

        assert set(from_kwargs.supported_variables) == set(
            from_spec.supported_variables
        )

    def test_spec_and_keywords_together_raise(self, lattice_json):
        with pytest.raises(ValueError, match="one or the other"):
            build_cheetah_model(
                {"lattice": lattice_json}, feature="dup", energy=1e8
            )

    def test_no_spec_at_all_raises(self):
        with pytest.raises(ValueError, match="requires a spec"):
            build_cheetah_model(energy=1e8)

    def test_unknown_spec_field_raises(self, lattice_json):
        with pytest.raises(TypeError, match="not_a_field"):
            build_cheetah_model(lattice=lattice_json, not_a_field=1, energy=1e8)

    def test_live_segment_is_used_directly(self, segment):
        model = build_cheetah_model(CheetahModelSpec(lattice=segment), energy=1e8)

        assert model.simulator.segment is segment

    def test_lattice_path_is_loaded(self, lattice_path):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=str(lattice_path)), energy=1e8
        )

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables

    def test_env_var_route(self, lattice_path, monkeypatch):
        monkeypatch.setenv("TEST_LATTICE_ROOT", str(lattice_path.parent))
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice_env_var="TEST_LATTICE_ROOT",
                lattice_relpath=lattice_path.name,
            ),
            energy=1e8,
        )

        assert "QUAD:IN10:511:BCTRL" in model.supported_variables

    def test_spec_json_file(self, lattice_json, tmp_path):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({"lattice": lattice_json}))

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
            build_cheetah_model(CheetahModelSpec(), energy=1e8)

    def test_non_lattice_type_raises(self):
        with pytest.raises(TypeError, match="lattice"):
            build_cheetah_model(CheetahModelSpec(lattice=42), energy=1e8)


class TestControlNames:
    def test_elements_without_alias_are_skipped(self):
        # No `alias` in metadata: the element is skipped with a warning rather
        # than failing the whole build.
        segment = Segment(
            [
                Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
                Drift(name="D1", length=torch.tensor(1.0)),
            ],
            name="TEST",
        )

        with pytest.warns(UserWarning, match="alias"):
            model = build_cheetah_model(CheetahModelSpec(lattice=segment), energy=1e8)

        assert model.supported_variables == {}

    def test_shipped_profmon_config_is_loadable(self):
        config = load_profmon_config("facet2_profmon_info.yaml")

        assert config["PR10571"]["name"] == "PROF:IN10:571"


class TestScreenGeometry:
    def test_lattice_geometry_is_kept_by_default(self, lattice_json, profmon_path):
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice=lattice_json,
                profmon_config_filename=str(profmon_path),
            ),
            energy=1e8,
        )

        # Geometry is left untouched unless `screen_geometry=True`: its shape
        # axis order is not consistent across profmon tables.
        assert tuple(model.simulator.segment.PR1.resolution) == (64, 48)
        assert model.supported_variables["OTRS:IN10:571:Image:ArrayData"].shape == (
            64,
            48,
        )

    def test_screen_geometry_opt_in(self, lattice_json, profmon_path):
        model = build_cheetah_model(
            CheetahModelSpec(
                lattice=lattice_json,
                profmon_config_filename=str(profmon_path),
                screen_geometry=True,
            ),
            energy=1e8,
        )
        screen = model.simulator.segment.PR1

        assert tuple(screen.resolution) == (128, 96)
        # Config is in micrometers, Cheetah in meters.
        assert float(screen.pixel_size[0]) == pytest.approx(11.1732e-6)
        assert model.supported_variables["OTRS:IN10:571:Image:ArrayData"].shape == (
            128,
            96,
        )
        # RESOLUTION reads back in micrometers, matching the config.
        assert float(model.get("OTRS:IN10:571:RESOLUTION")) == pytest.approx(
            11.1732, rel=1e-6
        )
        assert tuple(model.get("OTRS:IN10:571:Image:ArrayData").shape) == (128, 96)

    def test_screen_geometry_without_profmon_raises(self, lattice_json):
        with pytest.raises(ValueError, match="screen_geometry"):
            build_cheetah_model(
                CheetahModelSpec(lattice=lattice_json, screen_geometry=True),
                energy=1e8,
            )

    def test_unconfigured_screen_keeps_lattice_geometry(self, segment):
        updated = apply_screen_geometry(segment, {"SOMETHING_ELSE": {"shape": [8, 8]}})

        assert updated == []
        assert tuple(segment.PR1.resolution) == (64, 48)


class TestIncomingBeam:
    def test_energy_builds_placeholder_beam(self, lattice_json):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json), energy=1.25e8
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
            CheetahModelSpec(lattice=lattice_json),
            initial_beam_distribution=beam,
        )

        # The simulator keeps its own copy, so compare contents rather than identity.
        initial_beam = model.simulator.initial_beam_distribution
        assert float(initial_beam.energy) == float(beam.energy)
        assert initial_beam.particles.shape == beam.particles.shape

    def test_no_beam_and_no_energy_raises(self, lattice_json):
        with pytest.raises(ValueError, match="incoming beam"):
            build_cheetah_model(CheetahModelSpec(lattice=lattice_json))


class TestElementAttrMapping:
    def test_explicit_mapping_limits_variables(self, lattice_json):
        model = build_cheetah_model(
            CheetahModelSpec(lattice=lattice_json),
            energy=1e8,
            element_attr_mapping={"Quadrupole": {"BCTRL": "QuadrupoleBCTRLVariable"}},
        )

        quad_pvs = {
            name
            for name in model.supported_variables
            if name.startswith("QUAD:IN10:511")
        }
        assert quad_pvs == {"QUAD:IN10:511:BCTRL"}
