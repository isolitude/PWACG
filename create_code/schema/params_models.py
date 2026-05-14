#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class RunConfig(BaseModel, frozen=True):
    total_gpu_id: tuple[int, ...] = ()
    processes_gpus: Optional[int] = None
    max_processes: Optional[int] = None
    max_processes_memory: Optional[float] = None
    thread_gpus: Optional[int] = None
    threads_in_one_gpu: Optional[int] = None


class DataConfig(BaseModel, frozen=True):
    data_slices: Optional[int] = None
    mc_slices: Optional[int] = None
    mini_run: Optional[int] = None


class ModuleParams(BaseModel, frozen=True):
    run_config: RunConfig
    data_config: DataConfig


class DrawSwitch(BaseModel, frozen=True):
    likelihood: bool
    weight: bool
    pull: bool
    mods: bool


class PullOption(BaseModel, frozen=True):
    bin: int
    min: float
    max: float


class WeightOption(BaseModel, frozen=True):
    bin: int
    mods_num: int


class DrawConfig(BaseModel, frozen=True):
    switch: DrawSwitch
    pull_option: PullOption
    weight_option: WeightOption


class CacheTensorEntry(BaseModel, frozen=True):
    data: int
    mc: int


class ParametersFile(BaseModel, frozen=True):
    parameters: dict[str, ModuleParams]
    draw_config: DrawConfig
    CacheTensor: dict[str, CacheTensorEntry]
