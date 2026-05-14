"""Pydantic v2 schemas for PWACG configuration files.

Three top-level JSON inputs:
- pwa_info_*.json   -> PWAInfo (pwa_models)
- generator_*.json  -> GeneratorConfig (generator_models)
- parameters.json   -> ParametersFile (params_models)
"""
from .pwa_models import (
    PWAInfo,
    ModInfo,
    PropSpec,
    PropGroup,
    SbcSpec,
    ArgSpec,
)
from .generator_models import (
    GeneratorConfig,
    FitArtifact,
    DrawArtifact,
    AnnexInfo,
    FitInfo,
    PullInfo,
    DrawInfo,
    CombineInfo,
    JsonPwa,
)
from .params_models import (
    ParametersFile,
    ModuleParams,
    RunConfig,
    DataConfig,
    DrawConfig,
    DrawSwitch,
    PullOption,
    WeightOption,
    CacheTensorEntry,
)

__all__ = [
    "PWAInfo",
    "ModInfo",
    "PropSpec",
    "PropGroup",
    "SbcSpec",
    "ArgSpec",
    "GeneratorConfig",
    "FitArtifact",
    "DrawArtifact",
    "AnnexInfo",
    "FitInfo",
    "PullInfo",
    "DrawInfo",
    "CombineInfo",
    "JsonPwa",
    "ParametersFile",
    "ModuleParams",
    "RunConfig",
    "DataConfig",
    "DrawConfig",
    "DrawSwitch",
    "PullOption",
    "WeightOption",
    "CacheTensorEntry",
]
