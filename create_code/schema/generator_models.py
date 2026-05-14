#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations
from typing import Optional, Union
from pydantic import BaseModel


class FitArtifact(BaseModel, frozen=True):
    CodeTemplate: str
    CodeScript: str
    RunTemplate: str
    RunScript: str
    ResultFile: str = ""


class DrawArtifact(BaseModel, frozen=True):
    CodeTemplate: str
    CodeScript: tuple[str, ...]
    RunTemplate: str
    RunScript: str
    ResultFile: tuple[str, ...] = ()
    LassoResultFile: tuple[str, ...] = ()


class TotalFrac(BaseModel, frozen=True):
    kk: Optional[float] = None
    pipi: Optional[float] = None


class FitInfo(BaseModel, frozen=True):
    use_weight: bool
    write: bool
    Cycles: int
    total_frac: dict[str, float]
    boundary: bool
    lambda_tfc: float
    random: bool


class PullInfo(BaseModel, frozen=True):
    use_weight: bool
    write: bool


class DrawInfo(BaseModel, frozen=True):
    use_weight: bool
    write: bool
    frac_cut: float


class CombineInfo(BaseModel, frozen=True):
    tag: tuple[str, ...]


class AnnexInfo(BaseModel, frozen=True):
    pull: PullInfo
    fit: FitInfo
    draw: DrawInfo
    merge: str
    combine: CombineInfo


class JsonPwa(BaseModel, frozen=True):
    fit: tuple[str, ...]
    draw: tuple[str, ...]


class GeneratorConfig(BaseModel, frozen=True):
    id: str
    jinja_fit_info: dict[str, FitArtifact]
    jinja_draw_info: dict[str, DrawArtifact]
    json_pwa: JsonPwa
    annex_info: AnnexInfo
