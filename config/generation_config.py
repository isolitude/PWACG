# PWA代码生成器配置示例
# 这个配置定义了如何生成特定的拟合脚本

generation_config = {
    "template_file": "templates/fit_template_hvp.py",
    "output_file": "rendered_scripts/fit_kk_generated.py",
    
    # 模板块替换配置
    "template_blocks": {
        "PARAMETER_EXTRACTION": {
            "type": "resonance_parameters",
            "resonances": [
                {
                    "name": "f980",
                    "type": "flatte980", 
                    "args_start": 0,
                    "n_components": 2
                },
                {
                    "name": "f0_1710",
                    "type": "BW",
                    "args_start": 5,
                    "n_components": 2
                },
                {
                    "name": "f1270", 
                    "type": "flatte1270",
                    "args_start": 11,
                    "n_components": 5
                },
                {
                    "name": "f2",
                    "type": "BW_multi",
                    "args_start": 23,
                    "n_components": 5,
                    "n_masses": 3
                }
            ]
        },
        
        "DATA_INITIALIZATION": {
            "type": "data_members",
            "data_types": ["data", "mc", "truth"],
            "channels": ["phi_kk", "f_kk", "phif0_kk", "phif2_kk"],
            "additional": ["wt_data_kk"]
        },
        
        "DATA_LIKELIHOOD_CALCULATION": {
            "type": "amplitude_calculation",
            "amplitudes": [
                {
                    "name": "data_phif0_kk_BW_flatte980",
                    "function": "calculate_BW_flatte980",
                    "resonance1": "phi",
                    "resonance2": "f980",
                    "data_source": "data",
                    "wave": "phif0_kk"
                },
                {
                    "name": "data_phif0_kk_BW_BW", 
                    "function": "calculate_BW_BW",
                    "resonance1": "phi",
                    "resonance2": "f0",
                    "data_source": "data",
                    "wave": "phif0_kk"
                },
                {
                    "name": "data_phif2_kk_BW_flatte1270",
                    "function": "calculate_BW_flatte1270",
                    "resonance1": "phi", 
                    "resonance2": "f1270",
                    "data_source": "data",
                    "wave": "phif2_kk"
                },
                {
                    "name": "data_phif2_kk_BW_BW",
                    "function": "calculate_BW_BW",
                    "resonance1": "phi",
                    "resonance2": "f2",
                    "data_source": "data", 
                    "wave": "phif2_kk"
                }
            ]
        },
        
        "CONSTRAINT_CALCULATION": {
            "type": "lasso_constraint",
            "constraint_target": 1.03,
            "constraint_strength": "self.constraint_strength", 
            "penalty_factor": 10000.0,
            "amplitudes": [
                "lasso_data_phif0_kk_BW_flatte980",
                "lasso_data_phif0_kk_BW_BW", 
                "lasso_data_phif2_kk_BW_flatte1270",
                "lasso_data_phif2_kk_BW_BW"
            ]
        },
        
        "MC_LIKELIHOOD_CALCULATION": {
            "type": "mc_integration",
            "amplitudes": [
                "calculate_BW_flatte980",
                "calculate_BW_BW", 
                "calculate_BW_flatte1270",
                "calculate_BW_BW"
            ],
            "aggregation": "mean"
        },
        
        "INITIAL_PARAMETERS": {
            "type": "parameter_array",
            "source": "config/fit_kk_config.toml",
            "float_indices_source": "config/fit_kk_config.toml"
        },
        
        "OPTIMIZATION_EXECUTION": {
            "type": "optimization_method",
            "method": "Newton-CG",
            "options": {
                "disp": False,
                "xtol": 1e-8
            },
            "use_hvp": True,
            "callback": True
        }
    },
    
    # 导入配置
    "imports": {
        "common_modules": [
            "setup_logging", "load_data", "normalize_data", "prepare_data_for_jax"
        ],
        "resonance_functions": [
            "calculate_BW_flatte980", "calculate_BW_BW", "calculate_BW_flatte1270",
            "lasso_calculate_BW_flatte980", "lasso_calculate_BW_BW", "lasso_calculate_BW_flatte1270"
        ],
        "additional_imports": []
    },
    
    # 生成选项
    "generation_options": {
        "preserve_comments": True,
        "add_generation_timestamp": True,
        "add_config_reference": True,
        "output_format": "python",
        "include_type_hints": False
    }
}

# 物理配置
physics_config = {
    "fixed_parameters": {
        "phi_mass": 1.02,
        "phi_width": 0.004
    },
    
    "channels": {
        "kk": {
            "data_files": {
                "real_data": ["phif0_kk.npy", "phif2_kk.npy", "phi_kk.npy", "f_kk.npy"],
                "mc_truth": ["phif0_kk.npy", "phif2_kk.npy", "phi_kk.npy", "f_kk.npy"], 
                "weight": ["weight_kk.npy"]
            },
            "truth_sample_size": 150000
        }
    },
    
    "constraints": {
        "lasso_constraint": {
            "enabled": True,
            "target_value": 1.03,
            "penalty_factor": 10000.0,
            "default_strength": 0.0
        }
    },
    
    "optimization": {
        "likelihood_formula": "data_likelihood + data_size * log(mc_likelihood)",
        "jax_config": {
            "enable_x64": True
        }
    }
}