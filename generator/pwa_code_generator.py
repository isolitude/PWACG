#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PWA代码生成器示例
根据配置和模板生成特定的拟合脚本
"""

import re
import os
from datetime import datetime
from config.generation_config import generation_config, physics_config

class PWACodeGenerator:
    """PWA代码生成器"""
    
    def __init__(self, config, physics_config):
        self.config = config
        self.physics_config = physics_config
        self.template_blocks = {}
        
    def load_template(self):
        """加载模板文件"""
        with open(self.config["template_file"], 'r', encoding='utf-8') as f:
            self.template_content = f.read()
        
        # 解析模板块
        self._parse_template_blocks()
    
    def _parse_template_blocks(self):
        """解析模板中的可替换块"""
        # 查找所有模板块
        pattern = r'# TEMPLATE_BLOCK: (\w+)\n(.*?)# END_TEMPLATE_BLOCK: \1'
        matches = re.findall(pattern, self.template_content, re.DOTALL)
        
        for block_name, block_content in matches:
            self.template_blocks[block_name] = block_content.strip()
    
    def generate_parameter_extraction(self, resonances):
        """生成参数提取代码"""
        code_lines = ["return {"]
        code_lines.append("    # 固定参数")
        code_lines.append("    'phi_mass': np.array([1.02]),")
        code_lines.append("    'phi_width': np.array([0.004]),")
        code_lines.append("")
        
        for resonance in resonances:
            name = resonance["name"]
            res_type = resonance["type"]
            start_idx = resonance["args_start"]
            n_comp = resonance["n_components"]
            
            code_lines.append(f"    # {name}参数")
            
            if res_type == "flatte980":
                code_lines.append(f"    'kk_{name}_mass': np.array([args[{start_idx}]]),")
                code_lines.append(f"    'kk_{name}_g_kk': np.array([args[{start_idx+1}]]),")
                code_lines.append(f"    'kk_{name}_rg': np.array([args[{start_idx+2}]]),")
                code_lines.append(f"    'kk_{name}_const': np.array([0.1, args[{start_idx+3}]]).reshape(-1, 2),")
                code_lines.append(f"    'kk_{name}_theta': np.array([0.1, args[{start_idx+4}]]).reshape(-1, 2),")
                
            elif res_type == "BW":
                code_lines.append(f"    'kk_{name}_mass': np.array([args[{start_idx}]]),")
                code_lines.append(f"    'kk_{name}_width': np.array([args[{start_idx+1}]]),")
                const_args = [f"args[{start_idx+2+i}]" for i in range(n_comp)]
                theta_args = [f"args[{start_idx+2+n_comp+i}]" for i in range(n_comp)]
                code_lines.append(f"    'kk_{name}_const': np.array([{', '.join(const_args)}]).reshape(-1, {n_comp}),")
                code_lines.append(f"    'kk_{name}_theta': np.array([{', '.join(theta_args)}]).reshape(-1, {n_comp}),")
                
            elif res_type == "BW_multi":
                n_masses = resonance.get("n_masses", 1)
                mass_args = [f"args[{start_idx+i}]" for i in range(n_masses)]
                width_args = [f"args[{start_idx+n_masses+i}]" for i in range(n_masses)]
                code_lines.append(f"    'kk_{name}_mass': np.array([{', '.join(mass_args)}]),")
                code_lines.append(f"    'kk_{name}_width': np.array([{', '.join(width_args)}]),")
                
                const_start = start_idx + 2 * n_masses
                theta_start = const_start + n_masses * n_comp
                const_args = [f"args[{const_start+i}]" for i in range(n_masses * n_comp)]
                theta_args = [f"args[{theta_start+i}]" for i in range(n_masses * n_comp)]
                code_lines.append(f"    'kk_{name}_const': np.array([{', '.join(const_args)}]).reshape(-1, {n_comp}),")
                code_lines.append(f"    'kk_{name}_theta': np.array([{', '.join(theta_args)}]).reshape(-1, {n_comp}),")
            
            code_lines.append("")
        
        code_lines.append("}")
        return "\n".join(code_lines)
    
    def generate_amplitude_calculations(self, amplitudes):
        """生成振幅计算代码"""
        code_lines = []
        for amp in amplitudes:
            func_name = amp["function"]
            res1 = amp["resonance1"]
            res2 = amp["resonance2"]
            data_src = amp["data_source"]
            wave = amp["wave"]
            
            # 构建函数调用
            if "flatte980" in func_name:
                params = f"params['{res1}_mass'], params['{res1}_width'], params['kk_{res2}_mass'], params['kk_{res2}_g_kk'], params['kk_{res2}_rg'], params['kk_{res2}_const'], params['kk_{res2}_theta']"
            elif "flatte1270" in func_name:
                params = f"params['{res1}_mass'], params['{res1}_width'], params['kk_{res2}_mass'], params['kk_{res2}_width'], params['kk_{res2}_const'], params['kk_{res2}_theta']"
            else:  # BW_BW
                params = f"params['{res1}_mass'], params['{res1}_width'], params['kk_{res2}_mass'], params['kk_{res2}_width'], params['kk_{res2}_const'], params['kk_{res2}_theta']"
            
            data_params = f"self.{data_src}_{wave.split('_')[0]}_kk, self.{data_src}_{wave.split('_')[1]}_kk, self.{data_src}_{wave}"
            
            code_lines.append(f"{amp['name']} = {func_name}(")
            code_lines.append(f"    {params},")
            code_lines.append(f"    {data_params}")
            code_lines.append(")")
            code_lines.append("")
        
        return "\n".join(code_lines)
    
    def generate_code(self):
        """生成完整代码"""
        generated_content = self.template_content
        
        # 替换各个模板块
        for block_name, block_config in self.config["template_blocks"].items():
            if block_name not in self.template_blocks:
                continue
                
            if block_config["type"] == "resonance_parameters":
                new_content = self.generate_parameter_extraction(block_config["resonances"])
            elif block_config["type"] == "amplitude_calculation":
                new_content = self.generate_amplitude_calculations(block_config["amplitudes"])
            else:
                # 保持原内容
                new_content = self.template_blocks[block_name]
            
            # 替换模板块
            pattern = f'# TEMPLATE_BLOCK: {block_name}\\n.*?# END_TEMPLATE_BLOCK: {block_name}'
            replacement = f'# GENERATED_BLOCK: {block_name}\n{new_content}\n        # END_GENERATED_BLOCK: {block_name}'
            generated_content = re.sub(pattern, replacement, generated_content, flags=re.DOTALL)
        
        # 添加生成信息
        if self.config["generation_options"]["add_generation_timestamp"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_comment = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PWA拟合脚本 - 自动生成版本
生成时间: {timestamp}
模板文件: {self.config["template_file"]}
配置文件: config/generation_config.py
"""

'''
            generated_content = header_comment + generated_content[generated_content.find('\n', 50)+1:]
        
        return generated_content
    
    def save_generated_code(self, content):
        """保存生成的代码"""
        output_file = self.config["output_file"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"代码生成完成: {output_file}")

def main():
    """主函数"""
    generator = PWACodeGenerator(generation_config, physics_config)
    
    # 加载模板
    generator.load_template()
    
    # 生成代码
    generated_code = generator.generate_code()
    
    # 保存代码
    generator.save_generated_code(generated_code)
    
    print("PWA代码生成完成!")

if __name__ == "__main__":
    main()