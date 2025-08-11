#!/usr/bin/env python3
# coding: utf-8
"""
LLM Agent 基础类
用于替代 Jinja2 模板功能
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseAgent(ABC):
    """LLM Agent 基础抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def generate_code(self, template_data: Dict[str, Any]) -> str:
        """
        生成代码的抽象方法
        
        Args:
            template_data: 模板数据字典
            
        Returns:
            生成的代码字符串
        """
        pass
    
    @abstractmethod
    def validate_output(self, generated_code: str) -> bool:
        """
        验证生成代码的抽象方法
        
        Args:
            generated_code: 生成的代码
            
        Returns:
            验证是否通过
        """
        pass
    
    def save_output(self, code: str, output_path: str) -> bool:
        """
        保存生成的代码到文件
        
        Args:
            code: 生成的代码
            output_path: 输出文件路径
            
        Returns:
            保存是否成功
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            self.logger.info(f"代码已保存到: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存文件失败: {e}")
            return False


class CodeGenerationError(Exception):
    """代码生成异常"""
    pass


class ValidationError(Exception):
    """验证异常"""
    pass