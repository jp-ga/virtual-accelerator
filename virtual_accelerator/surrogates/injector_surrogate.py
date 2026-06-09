import torch

from virtual_accelerator.surrogates.beam_output import BeamOutputModel
from virtual_accelerator.surrogates.utils import compute_covariance_matrix

from lcls_cu_inj_model import load_model


class InjectorSurrogate(BeamOutputModel):
    """
    Custom wrapper class for the LCLS injector surrogate model, which
    computes the covariance matrix from the surrogate's scalar beam parameters.

    Hardcoded to use the LCLS injector surrogate model which dumps beam at OTR2
    """

    def __init__(self, **kwargs):
        super().__init__(load_model(), **kwargs)
        self.p0c = 135.0e6  # eV/c

    def update_state(self):
        """Wrap the generic `update_state` method to calculate the custom covariance matrix"""
        # Update cache with surrogate outputs
        self._cache.update(
            self.surrogate.get(list(self.surrogate.supported_variables.keys()))
        )

        # Compute the covariance matrix from surrogate outputs and store in cache
        cov = compute_covariance_matrix(self._cache, self.p0c)
        self._cache["covariance_matrix"] = torch.from_numpy(cov)

        # Generate the output beam ParticleGroup
        self._generate_output_beam()
