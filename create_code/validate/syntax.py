#!/usr/bin/env python3
# coding: utf-8
"""Syntax validation: ast.parse check for LLM-generated Python code."""
from __future__ import annotations
import ast
from dataclasses import dataclass


@dataclass
class SyntaxResult:
    passed: bool
    error: str = ""
    error_line: int = 0


def check(file_content: str) -> SyntaxResult:
    """Check Python syntax via ast.parse."""
    try:
        ast.parse(file_content)
        return SyntaxResult(passed=True)
    except SyntaxError as e:
        return SyntaxResult(
            passed=False,
            error=str(e.msg),
            error_line=e.lineno or 0,
        )
