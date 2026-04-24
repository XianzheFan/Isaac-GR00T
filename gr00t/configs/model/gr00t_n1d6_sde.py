from dataclasses import dataclass

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from . import register_model_config


@dataclass
class Gr00tN1d6SDEConfig(Gr00tN1d7Config):
    """Configuration for Gr00tN1d6 with SDE (Stochastic Differential Equation) sampling.

    Extends the base config with SDE-specific parameters for Euler-Maruyama inference.
    The model weights and training are identical to the ODE version; only the inference
    sampling loop changes from deterministic Euler to stochastic Euler-Maruyama.
    """

    model_type: str = "Gr00tN1d6SDE"

    # SDE-specific parameters
    noise_method: str = "flow_sde"
    noise_level: float = 0.5  # Controls diffusion noise strength during SDE sampling


register_model_config("GrootN1d6SDE", Gr00tN1d6SDEConfig)
