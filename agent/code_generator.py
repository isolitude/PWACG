#!/usr/bin/env python3
# coding: utf-8
"""
代码生成器模块
使用 LLM 和 Function Calling 来生成严格格式的代码
"""

import json
import ast
from typing import Dict, Any, List, Optional
from .base import BaseAgent, CodeGenerationError, ValidationError
from .openai_client import OpenAIClient
from .easytrans_client import EasyTransClient


class CodeGenerator(BaseAgent):
    """使用 LLM 的代码生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 检查使用哪种API
        self.api_provider = config.get('api_provider', 'openai').lower()
        
        if self.api_provider == 'easytrans':
            self.client = EasyTransClient(
                api_key=config.get('api_key') or config.get('easytrans_api_key'),
                base_url=config.get('base_url') or config.get('easytrans_base_url')
            )
            self.model = config.get('model', 'gemini-2.5-pro')
        else:  # 默认使用 OpenAI
            self.client = OpenAIClient(
                api_key=config.get('api_key') or config.get('openai_api_key'),
                base_url=config.get('base_url') or config.get('openai_base_url')
            )
            self.model = config.get('model', 'gpt-3.5-turbo')
    
    def _get_code_generation_functions(self) -> List[Dict[str, Any]]:
        """获取代码生成的函数定义"""
        return [
            {
                "name": "generate_python_code",
                "description": "生成 Python 代码",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "生成的完整 Python 代码"
                        },
                        "imports": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "需要的导入语句列表"
                        },
                        "description": {
                            "type": "string",
                            "description": "代码功能描述"
                        },
                        "main_function": {
                            "type": "string",
                            "description": "主要函数名称"
                        }
                    },
                    "required": ["code", "imports", "description"]
                }
            }
        ]
    
    def _build_prompt(self, template_data: Dict[str, Any], code_type: str) -> str:
        """构建生成代码的提示"""
        base_prompt = f"""
你是一个专业的 Python 代码生成器，需要根据提供的配置数据生成 {code_type} 类型的代码。

配置数据：
{json.dumps(template_data, indent=2, ensure_ascii=False)}

要求：
1. 生成的代码必须是完整的、可执行的 Python 脚本
2. 代码风格要整洁，遵循 PEP8 规范
3. 包含必要的导入语句
4. 添加适当的注释和文档字符串
5. 处理可能的异常情况
6. 使用提供的配置数据填充模板变量

请使用 generate_python_code 函数返回生成的代码。
"""
        
        if code_type == "fit":
            base_prompt += """
特殊要求（fit 类型）：
- 包含数据拟合相关的逻辑
- 使用多进程处理
- 包含日志记录功能
- 处理随机种子
"""
        elif code_type == "draw":
            base_prompt += """
特殊要求（draw 类型）：
- 包含数据可视化相关的逻辑
- 支持多种图表类型
- 处理结果文件读取
- 包含图片保存功能
"""
        
        return base_prompt
    
    def _call_easytrans_api(self, messages: List[Dict[str, str]], prompt: str) -> Dict[str, Any]:
        """
        调用极易云 API - 根据模型类型自动选择合适的 API
        
        Args:
            messages: 消息列表
            prompt: 提示文本
            
        Returns:
            API 响应结果
        """
        try:
            # Claude 系列模型使用 Messages API
            if 'claude' in self.model.lower():
                self.logger.info(f"检测到 Claude 模型 {self.model}，使用 Messages API")
                response = self.client.messages(
                    messages=messages,
                    model=self.model,
                    max_tokens=4000,  # Claude 需要设置 max_tokens
                    temperature=0.1
                )
                
                if self.client.validate_response(response):
                    return response
            
            # O3 系列模型优先使用 Responses API
            elif 'o3' in self.model.lower():
                self.logger.info(f"检测到 O3 模型 {self.model}，使用 Responses API")
                response = self.client.responses(
                    input_text=prompt,
                    model=self.model
                )
                
                if self.client.validate_response(response):
                    return response
            
            else:
                # 其他模型（GPT、Gemini）使用对话补全 API
                self.logger.info(f"使用对话补全 API，模型: {self.model}")
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.1
                )
                
                if self.client.validate_response(response):
                    return response
            
            # 如果首选 API 失败，尝试备用方案
            self.logger.warning("首选 API 失败，尝试备用方案")
            
            # 备用方案1: 尝试 Responses API
            try:
                backup_model = self.model
                if 'claude' in self.model.lower():
                    backup_model = 'o3-pro-2025-06-10'  # Claude 降级到 O3
                elif 'gemini' in self.model.lower():
                    backup_model = 'o3-pro-2025-06-10'  # Gemini 降级到 O3
                
                response = self.client.responses(
                    input_text=prompt,
                    model=backup_model
                )
                
                if self.client.validate_response(response):
                    self.logger.info(f"备用方案成功，使用模型: {backup_model}")
                    return response
                    
            except Exception as backup_e:
                self.logger.warning(f"备用方案失败: {backup_e}")
            
            # 备用方案2: 尝试基础的对话补全 API
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    model='gemini-2.5-pro',  # 使用稳定的基础模型
                    temperature=0.1
                )
                
                if self.client.validate_response(response):
                    self.logger.info("使用基础模型成功")
                    return response
                    
            except Exception as final_e:
                self.logger.error(f"所有备用方案都失败: {final_e}")
            
            raise CodeGenerationError("所有 API 调用方案都失败")
            
        except Exception as e:
            self.logger.error(f"极易云 API 调用失败: {e}")
            raise CodeGenerationError(f"极易云 API 调用失败: {e}")
    
    def generate_code(self, template_data: Dict[str, Any], code_type: str = "generic") -> str:
        """
        生成代码
        
        Args:
            template_data: 模板数据字典
            code_type: 代码类型 (fit, draw, generic 等)
            
        Returns:
            生成的代码字符串
        """
        try:
            prompt = self._build_prompt(template_data, code_type)
            
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # 根据API提供商调用不同的接口
            if self.api_provider == 'easytrans':
                # 极易云 API 调用
                response = self._call_easytrans_api(messages, prompt)
                generated_code = self.client.extract_content(response)
                if not generated_code:
                    raise CodeGenerationError("生成的代码为空")
                
                # 记录生成信息
                self.logger.info(f"代码生成成功，类型: {code_type}")
                arguments = {"imports": [], "description": f"Generated {code_type} code"}
            else:
                # OpenAI API 调用 (支持 Function Calling)
                functions = self._get_code_generation_functions()
                
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.1,
                    functions=functions,
                    function_call={"name": "generate_python_code"}
                )
                
                if not self.client.validate_response(response):
                    raise CodeGenerationError("LLM 响应无效")
                
                function_call = self.client.extract_function_call(response)
                if not function_call:
                    raise CodeGenerationError("未获取到函数调用")
                
                # 解析函数参数
                try:
                    arguments = json.loads(function_call['arguments'])
                except json.JSONDecodeError as e:
                    raise CodeGenerationError(f"解析函数参数失败: {e}")
                
                generated_code = arguments.get('code', '')
                if not generated_code:
                    raise CodeGenerationError("生成的代码为空")
            
            # 记录生成信息
            self.logger.info(f"代码生成成功，类型: {code_type}")
            self.logger.info(f"导入模块: {arguments.get('imports', [])}")
            self.logger.info(f"功能描述: {arguments.get('description', '')}")
            
            return generated_code
            
        except Exception as e:
            self.logger.error(f"代码生成失败: {e}")
            raise CodeGenerationError(f"代码生成失败: {e}")
    
    def validate_output(self, generated_code: str) -> bool:
        """
        验证生成的代码
        
        Args:
            generated_code: 生成的代码
            
        Returns:
            验证是否通过
        """
        try:
            # 语法检查
            ast.parse(generated_code)
            
            # 基本结构检查
            if not generated_code.strip():
                raise ValidationError("代码为空")
            
            # 检查是否包含基本的 Python 结构
            if 'import' not in generated_code and 'from' not in generated_code:
                self.logger.warning("代码中没有导入语句")
            
            if 'def ' not in generated_code and 'class ' not in generated_code:
                self.logger.warning("代码中没有函数或类定义")
            
            self.logger.info("代码验证通过")
            return True
            
        except SyntaxError as e:
            self.logger.error(f"代码语法错误: {e}")
            raise ValidationError(f"代码语法错误: {e}")
        except Exception as e:
            self.logger.error(f"代码验证失败: {e}")
            raise ValidationError(f"代码验证失败: {e}")
    
    def generate_and_save(
        self,
        template_data: Dict[str, Any],
        output_path: str,
        code_type: str = "generic"
    ) -> bool:
        """
        生成代码并保存到文件
        
        Args:
            template_data: 模板数据
            output_path: 输出文件路径
            code_type: 代码类型
            
        Returns:
            操作是否成功
        """
        try:
            # 生成代码
            generated_code = self.generate_code(template_data, code_type)
            
            # 验证代码
            if not self.validate_output(generated_code):
                return False
            
            # 保存代码
            return self.save_output(generated_code, output_path)
            
        except Exception as e:
            self.logger.error(f"生成并保存代码失败: {e}")
            return False


class PWACodeGenerator(CodeGenerator):
    """专门用于 PWA 项目的代码生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def generate_fit_code(self, template_data: Dict[str, Any]) -> str:
        """生成拟合代码"""
        return self.generate_code(template_data, "fit")
    
    def generate_draw_code(self, template_data: Dict[str, Any]) -> str:
        """生成绘图代码"""
        return self.generate_code(template_data, "draw")
    
    def generate_tensor_code(self, template_data: Dict[str, Any]) -> str:
        """生成张量计算代码"""
        return self.generate_code(template_data, "tensor")
    
    def batch_generate(
        self,
        template_configs: List[Dict[str, Any]],
        output_dir: str = "rendered_scripts"
    ) -> List[bool]:
        """
        批量生成代码
        
        Args:
            template_configs: 模板配置列表
            output_dir: 输出目录
            
        Returns:
            每个生成任务的成功状态列表
        """
        results = []
        
        for i, config in enumerate(template_configs):
            try:
                code_type = config.get('type', 'generic')
                output_file = f"{output_dir}/{code_type}_{i}.py"
                
                success = self.generate_and_save(
                    config.get('data', {}),
                    output_file,
                    code_type
                )
                results.append(success)
                
            except Exception as e:
                self.logger.error(f"批量生成第 {i} 个任务失败: {e}")
                results.append(False)
        
        return results