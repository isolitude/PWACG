#!/usr/bin/env python3
# coding: utf-8
"""Validate module — syntax, type, and structural checks for generated code."""
from .syntax import check as check_syntax, SyntaxResult
from .types import check as check_types, TypeResult
from .structural import check as check_structural, check_lh_methods, StructuralResult

__all__ = [
    "check_syntax",
    "check_types",
    "check_structural",
    "check_lh_methods",
    "SyntaxResult",
    "TypeResult",
    "StructuralResult",
]
