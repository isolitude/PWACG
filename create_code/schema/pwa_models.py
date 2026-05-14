#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, model_validator


class PropSpec(BaseModel, frozen=True):
    name: str
    paras: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def _coerce_paras(cls, data: dict) -> dict:
        if isinstance(data.get("paras"), list):
            data["paras"] = tuple(data["paras"])
        return data


class SbcSpec(BaseModel, frozen=True):
    phi: str
    f: str


class ArgSpec(BaseModel, frozen=True):
    value: float
    name: str
    fix: bool = False
    error: float = 0.0
    range: Optional[tuple[float, float]] = None
    binding: Optional[dict] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_range(cls, data: dict) -> dict:
        if isinstance(data.get("range"), list):
            data["range"] = tuple(data["range"])
        return data


class PropGroup(BaseModel, frozen=True):
    prop_phi: PropSpec
    prop_f: PropSpec


class ModInfo(BaseModel, frozen=True):
    mod: str
    amp: str
    prop: PropGroup
    Sbc: SbcSpec
    args: dict[str, ArgSpec]


class PWAInfo(BaseModel, frozen=True):
    mod_info: tuple[ModInfo, ...]
    external_binding: dict[str, dict] = {}

    @model_validator(mode="before")
    @classmethod
    def _coerce_mod_info(cls, data: dict) -> dict:
        if isinstance(data.get("mod_info"), list):
            data["mod_info"] = tuple(data["mod_info"])
        return data
