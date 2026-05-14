"""IR layer — intermediate representation between Schema and codegen."""
from .ir_models import (
    PWAIR,
    ParamTable,
    BindingEdge,
    BindingGraph,
    RangeConstraint,
    FuncInfo,
    SlitArg,
    LHEntry,
    ArgsIndexCollection,
    LassoMeta,
)
from .builder import build_ir

__all__ = [
    "PWAIR",
    "ParamTable",
    "BindingEdge",
    "BindingGraph",
    "RangeConstraint",
    "FuncInfo",
    "SlitArg",
    "LHEntry",
    "ArgsIndexCollection",
    "LassoMeta",
    "build_ir",
]
