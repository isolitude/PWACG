#!/usr/bin/env python3
# coding: utf-8
"""Structural validation: verify required methods/classes exist in generated code.

Checks driven by ArtifactSpec.required_methods and lh_coll tags.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field


@dataclass
class StructuralResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def check(file_content: str, *, required_methods: tuple[str, ...] = ()) -> StructuralResult:
    """Check that generated code has all required methods/classes."""
    errors: list[str] = []
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return StructuralResult(passed=True)  # caught by syntax check

    # Collect all function and method names
    func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.add(node.name)

    for method in required_methods:
        if method not in func_names:
            errors.append(f"Missing required method/function: '{method}'")

    return StructuralResult(passed=len(errors) == 0, errors=errors)


def check_lh_methods(file_content: str, *, tags: tuple[str, ...]) -> StructuralResult:
    """Check that for each lh_coll tag, the required 4 methods exist."""
    errors: list[str] = []
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return StructuralResult(passed=True)

    func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.add(node.name)

    for tag in tags:
        for prefix in ("data_likelihood_", "mc_likelihood_", "weight_", "wt_data_likelihood_"):
            expected = f"{prefix}{tag}"
            if expected not in func_names:
                errors.append(f"Missing method: {expected}")

    return StructuralResult(passed=len(errors) == 0, errors=errors)
