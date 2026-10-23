"""Shared assembly for Cheetah-backed LUME models.

Symmetric with :mod:`virtual_accelerator.bmad.factory`: a frozen
:class:`CheetahModelSpec` declares *where* a lattice's ingredients come from and
:func:`build_cheetah_model` turns one into a ``LUMECheetahModel``, so
lattice-specific entry points in :mod:`virtual_accelerator.models` shrink to a
spec literal instead of re-inlining the four assembly steps (load lattice,
resolve control names, build a simulator, instantiate action variables).

Two provenance routes, because two kinds of caller need different things:

- **On-disk**, like the Bmad factory: ``lattice_env_var`` + ``lattice_relpath``
  locate the lattice under a lattice repository.
- **Inline**: ``lattice`` takes a lattice-JSON *string* (or a live ``Segment``).
  Nothing is read from the environment, so a caller that has serialized a model
  can rebuild it without depending on external files -- what a self-contained
  checkpoint needs.

Control (PV) names are not supplied by this factory at all: each element's
control name is read from its own ``metadata["alias"]`` entry in the lattice
JSON by :func:`virtual_accelerator.cheetah.variables.get_variables_from_segment`.
An element missing ``alias`` is skipped with a warning rather than failing the
whole build.

The spec itself is accepted in three interchangeable forms -- the dataclass, a
plain mapping of its fields, or those fields as keyword arguments -- so a caller
that stores the spec as JSON (no dataclass instance to pickle) or as flat
builder kwargs can call this factory without adapting its own record format.

Screen geometry
---------------
A screen's ``shape`` axis order is not consistent between profmon tables
(``facet2_profmon_info.yaml`` has ``[1392, 1040]``, ``cu_hxr_profmon_info.yaml``
has ``[1040, 1392]``), so applying it blindly can transpose a screen; the
lattice's own ``Screen.resolution`` is used instead unless
``screen_geometry=True`` requests the profmon config's ``shape``/``pixel_size``
explicitly, via ``profmon_config_filename``.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from virtual_accelerator.utils.optional_dependencies import import_optional

logger = logging.getLogger(__name__)

_REQUIRED_MODULES = ("torch", "cheetah", "lume_cheetah.model", "lume_torch.variables")


@dataclass(frozen=True)
class CheetahModelSpec:
    """Where a Cheetah model's ingredients come from.

    The lattice is supplied by exactly one of two routes: ``lattice`` (inline) or
    ``lattice_env_var`` + ``lattice_relpath`` (on disk). Control (PV) names are
    not part of the spec: each element supplies its own via
    ``metadata["alias"]`` in the lattice JSON.

    Attributes
    ----------
    feature : str
        Human-readable name of the model, used in the missing-optional-dependency
        error message.
    lattice : str | Segment | None
        Inline lattice: a lattice-JSON string, a path to a lattice JSON, or an
        already-constructed Cheetah ``Segment``. A string is treated as JSON text
        when it starts with ``{`` and as a path otherwise.
    lattice_env_var : str | None
        Environment variable holding the lattice repository root (e.g.
        ``"LCLS_LATTICE"``). Used to resolve ``lattice_relpath``.
    lattice_relpath : str | None
        Path of the lattice JSON relative to the lattice repository root (e.g.
        ``"cheetah/nc_hxr.json"``).
    profmon_config_filename : str | None
        Profmon config in ``virtual_accelerator/utils`` (e.g.
        ``"facet2_profmon_info.yaml"``) supplying screen geometry
        (``shape``/``pixel_size``) when ``screen_geometry=True``.
    screen_geometry : bool
        When ``True``, initialize each screen's ``resolution`` and
        ``pixel_size`` from the profmon config, overwriting the lattice values.
        Off by default -- see this module's docstring on axis order.
    """

    feature: str = "Cheetah model"
    lattice: Any = None
    lattice_env_var: str | None = None
    lattice_relpath: str | None = None
    profmon_config_filename: str | None = None
    screen_geometry: bool = False


def _check_optional_modules(module_names, feature: str, extra: str) -> None:
    """Validate all optional modules for a feature in a single gate check."""
    for module_name in module_names:
        import_optional(module_name, feature=feature, extra=extra)


def _coerce_spec(spec: "CheetahModelSpec | Mapping[str, Any]") -> CheetahModelSpec:
    """Accept a spec as a dataclass or as a plain mapping of its fields.

    The mapping form lets a caller keep a spec in JSON (no dataclass instance to
    pickle), which is what makes a serialized model rebuildable from a
    JSON-only record.
    """
    if isinstance(spec, CheetahModelSpec):
        return spec
    return CheetahModelSpec(**spec)


def _lattice_root(spec: CheetahModelSpec) -> str | None:
    """Resolve the lattice repository root from the environment, if configured."""
    if spec.lattice_env_var is None:
        return None

    root = os.environ.get(spec.lattice_env_var)
    if root is None:
        raise ValueError(
            f"{spec.feature} requires the {spec.lattice_env_var} environment "
            f"variable to be set, or an inline lattice."
        )
    return root


def _segment_from_lattice_json_text(lattice_json: str, dtype):
    """Build a ``Segment`` from lattice-JSON *text*.

    Cheetah only loads lattices from a path, so the text is round-tripped through
    a temporary file. Deliberately uses that public loader rather than Cheetah's
    internal dict-level parser.
    """
    from cheetah.accelerator import Segment

    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(lattice_json)
        return Segment.from_lattice_json(tmp_path, dtype=dtype)
    finally:
        os.remove(tmp_path)


def _resolve_segment(spec: CheetahModelSpec, dtype):
    """Load the lattice named by ``spec`` into a Cheetah ``Segment``."""
    from cheetah.accelerator import Segment

    if spec.lattice is not None:
        if isinstance(spec.lattice, Segment):
            return spec.lattice
        if not isinstance(spec.lattice, str):
            raise TypeError(
                f"{spec.feature}: 'lattice' must be a lattice-JSON string, a path, "
                f"or a cheetah Segment; got {type(spec.lattice).__name__}."
            )
        if spec.lattice.lstrip().startswith("{"):
            return _segment_from_lattice_json_text(spec.lattice, dtype)
        return Segment.from_lattice_json(spec.lattice, dtype=dtype)

    if spec.lattice_relpath is None:
        raise ValueError(
            f"{spec.feature} has no lattice: set 'lattice' (inline JSON, a path, "
            f"or a Segment) or both 'lattice_env_var' and 'lattice_relpath'."
        )

    root = _lattice_root(spec)
    if root is None:
        raise ValueError(
            f"{spec.feature} sets 'lattice_relpath' without 'lattice_env_var'; "
            f"the lattice repository root is unknown."
        )
    return Segment.from_lattice_json(
        os.path.join(root, spec.lattice_relpath), dtype=dtype
    )


def load_profmon_config(filename: str) -> dict[str, dict[str, Any]]:
    """Load a profmon config shipped in ``virtual_accelerator/utils``.

    Parameters
    ----------
    filename : str
        File name (e.g. ``"facet2_profmon_info.yaml"``), or an absolute path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Screen element name -> ``{"name", "shape", "pixel_size"}``.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / "utils" / filename
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_screen_geometry(
    segment,
    profmon_config: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Initialize screen geometry in ``segment`` from a profmon config.

    Sets each matching ``Screen``'s ``resolution`` (pixels) and ``pixel_size``
    (converted from the config's micrometers to Cheetah's meters), overwriting
    the lattice values, and clears the screen's cached reading so the new
    resolution takes effect.

    The config's ``shape`` axis order is not consistent across tables, so this is
    opt-in (``CheetahModelSpec.screen_geometry``) rather than automatic.

    Parameters
    ----------
    segment : Segment
        Cheetah segment to mutate in place.
    profmon_config : Mapping[str, Mapping[str, Any]]
        Screen element name -> ``{"shape", "pixel_size"}``, ``pixel_size`` in
        micrometers.

    Returns
    -------
    list[str]
        Names of the screens that were updated.
    """
    import torch
    from cheetah.accelerator import Screen

    updated: list[str] = []
    for element in segment.elements:
        if not isinstance(element, Screen):
            continue

        screen = profmon_config.get(element.name.split("#", 1)[0].upper())
        if screen is None:
            screen = profmon_config.get(element.name.split("#", 1)[0])
        if screen is None:
            logger.warning(
                "Screen %s is missing from the profmon configuration; keeping the "
                "lattice's resolution and pixel size.",
                element.name,
            )
            continue

        if "shape" in screen:
            element.resolution = tuple(int(n) for n in screen["shape"])
        if "pixel_size" in screen:
            element.pixel_size = torch.full_like(
                element.pixel_size, float(screen["pixel_size"]) * 1e-6
            )
        # `resolution` sizes the cached reading buffer; drop it so the next read
        # is recomputed at the new geometry.
        element.set_read_beam(None)
        updated.append(element.name)

    return updated


def build_cheetah_model(
    spec: "CheetahModelSpec | Mapping[str, Any] | None" = None,
    *,
    energy: float | None = None,
    initial_beam_distribution=None,
    initial_particle_group=None,
    element_attr_mapping: Mapping[str, Any] | None = None,
    dtype=None,
    **spec_fields,
):
    """Build a lattice-specific ``LUMECheetahModel`` from a shared implementation.

    Parameters
    ----------
    spec : CheetahModelSpec | Mapping[str, Any], optional
        Where the lattice comes from. A plain mapping of the dataclass's fields
        is accepted so a spec can be stored as JSON. May be omitted in favour of
        ``**spec_fields``.
    energy : float, optional
        Reference momentum p0c [eV/c], used only to build a placeholder incoming
        beam when neither ``initial_beam_distribution`` nor
        ``initial_particle_group`` is given.
    initial_beam_distribution : Beam, optional
        Incoming beam distribution.
    initial_particle_group : ParticleGroup, optional
        openPMD-beamphysics particle group, converted by the simulator.
    element_attr_mapping : Mapping[str, Any], optional
        Element type -> PV suffix -> variable class mapping. Defaults to the
        package's SLAC variable configuration.
    dtype : torch.dtype, optional
        Data type for the lattice elements. Defaults to Cheetah's own default.
    **spec_fields
        :class:`CheetahModelSpec` fields passed individually, in place of
        ``spec`` -- e.g. ``build_cheetah_model(lattice=..., energy=...)``. Lets
        a caller whose stored recipe is flat keyword arguments splat it
        straight in (``build_cheetah_model(**config, energy=energy)``) without
        reshaping the record. An unrecognized field
        raises ``TypeError``.

    Returns
    -------
    LUMECheetahModel
        Model wired to action variables for every supported element.

    Notes
    -----
    The incoming beam is optional because the beam is not always the caller's
    input: a reconstruction fits a *trainable* beam through this model and
    replaces the distribution on every forward pass, so it only needs the
    lattice to build. Passing ``energy`` alone yields a single-particle
    placeholder at that reference momentum.
    """
    if spec_fields:
        if spec is not None:
            raise ValueError(
                f"build_cheetah_model got both a 'spec' and spec fields as keyword "
                f"arguments ({', '.join(sorted(spec_fields))}); pass one or the other."
            )
        spec = spec_fields
    if spec is None:
        raise ValueError(
            "build_cheetah_model requires a spec: pass a CheetahModelSpec, a mapping "
            "of its fields, or those fields as keyword arguments."
        )

    spec = _coerce_spec(spec)

    _check_optional_modules(_REQUIRED_MODULES, feature=spec.feature, extra="cheetah")

    import torch
    from cheetah.particles import ParticleBeam
    from lume_cheetah.model import LUMECheetahModel
    from lume_cheetah.simulator import CheetahSimulator

    from virtual_accelerator.cheetah.variables import get_variables_from_segment

    segment = _resolve_segment(spec, dtype)

    profmon_config = (
        load_profmon_config(spec.profmon_config_filename)
        if spec.profmon_config_filename is not None
        else None
    )

    if spec.screen_geometry:
        if profmon_config is None:
            raise ValueError(
                f"{spec.feature} sets 'screen_geometry' without a "
                f"'profmon_config_filename'; there is no geometry to apply."
            )
        apply_screen_geometry(segment, profmon_config)

    if initial_beam_distribution is None and initial_particle_group is None:
        if energy is None:
            raise ValueError(
                f"{spec.feature} needs an incoming beam: pass "
                f"'initial_beam_distribution', 'initial_particle_group', or "
                f"'energy' for a placeholder beam."
            )
        initial_beam_distribution = ParticleBeam(
            torch.zeros(1, 7),
            energy=torch.tensor(energy, dtype=dtype or torch.float32),
        )

    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=initial_beam_distribution,
        initial_particle_group=initial_particle_group,
    )

    variables = get_variables_from_segment(
        segment,
        element_attr_mapping=element_attr_mapping,
    )

    return LUMECheetahModel(
        simulator=simulator,
        action_variables=list(variables.values()),
    )


def build_cheetah_model_from_json(path: str, **kwargs):
    """Build a model from a JSON file holding :class:`CheetahModelSpec` fields.

    Convenience for the inline route: the JSON object's keys are the spec's field
    names, so a lattice can be checked in as data rather than as a Python entry
    point.

    Parameters
    ----------
    path : str
        Path to a JSON object of ``CheetahModelSpec`` fields.
    **kwargs
        Forwarded to :func:`build_cheetah_model`.
    """
    with open(path) as f:
        return build_cheetah_model(json.load(f), **kwargs)
