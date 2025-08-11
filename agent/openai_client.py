#!/usr/bin/env python3
# coding: utf-8
"""
OpenAI API 客户端模块
用于与 OpenAI API 进行交互
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI


class OpenAIClient:
    """OpenAI API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: OpenAI API 密钥，如果为 None 则从环境变量读取
            base_url: API 基础 URL，支持自定义端点
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        if not self.api_key:
            raise ValueError("OpenAI API 密钥未设置，请设置 OPENAI_API_KEY 环境变量")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
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
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        发送聊天完成请求
        
        Args:
            messages: 消息列表
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大 token 数
            functions: 函数定义列表
            function_call: 函数调用设置
            
        Returns:
            API 响应结果
        """
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            if functions:
                kwargs["functions"] = functions
                
            if function_call:
                kwargs["function_call"] = function_call
            
            response = self.client.chat.completions.create(**kwargs)
            
            self.logger.info(f"API 调用成功，模型: {model}")
            return response.model_dump()
            
        except Exception as e:
            self.logger.error(f"API 调用失败: {e}")
            raise
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        验证 API 响应
        
        Args:
            response: API 响应
            
        Returns:
            验证是否通过
        """
        if not response.get('choices'):
            return False
            
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop' and not choice.get('message', {}).get('function_call'):
            return False
            
        return True
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        提取响应内容
        
        Args:
            response: API 响应
            
        Returns:
            提取的内容
        """
        if not response.get('choices'):
            return ""
            
        choice = response['choices'][0]
        message = choice.get('message', {})
        
        return message.get('content', '')
    
    def extract_function_call(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        提取函数调用
        
        Args:
            response: API 响应
            
        Returns:
            函数调用信息
        """
        if not response.get('choices'):
            return None
            
        choice = response['choices'][0]
        message = choice.get('message', {})
        
        return message.get('function_call')


class OpenAIError(Exception):
    """OpenAI API 异常"""
    pass