#!/usr/bin/env python3
# coding: utf-8
"""
极易云开放平台 API 客户端模块
支持对话补全、响应、消息、嵌入等多种API接口
"""

import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union


class EasyTransClient:
    """极易云开放平台 API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化极易云客户端
        
        Args:
            api_key: API 密钥，如果为 None 则从环境变量读取
            base_url: API 基础 URL
        """
        self.api_key = api_key or os.getenv('EASYTRANS_API_KEY') 
        self.base_url = base_url or os.getenv('EASYTRANS_BASE_URL', 'https://api.easytransnote.com/v1')
        
        if not self.api_key:
            raise ValueError("API 密钥未设置，请设置 EASYTRANS_API_KEY 或 OPENAI_API_KEY 环境变量")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        })
        
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
        model: str = "gemini-2.5-pro",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        对话补全 API - 创建对话式AI应用
        
        Args:
            messages: 对话历史数组
            model: 模型ID (如 gpt-4o, gemini-2.5-pro)
            temperature: 随机性控制 (0-2)
            max_tokens: 最大token数
            stream: 是否开启流式输出
            functions: 函数定义列表 (用于兼容OpenAI格式)
            function_call: 函数调用设置 (用于兼容OpenAI格式)
            
        Returns:
            API 响应结果
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            # "max_tokens": max_tokens or 4000  # 极易云需要显式设置 max_tokens
        }
        
        # 处理 function calling（如果极易云支持的话）
        if functions:
            payload["functions"] = functions
            self.logger.warning("极易云可能不支持 Function Calling，如有问题请改用 responses API")
        
        if function_call:
            payload["function_call"] = function_call
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            self.logger.info(f"对话补全 API 调用成功，模型: {model}")
            return result
            
        except Exception as e:
            self.logger.error(f"对话补全 API 调用失败: {e}")
            raise EasyTransError(f"对话补全 API 调用失败: {e}")
    
    def responses(
        self,
        input_text: str,
        model: str = "o3-pro-2025-06-10",
        stream: bool = False,
        background: bool = False
    ) -> Dict[str, Any]:
        """
        响应 API - 轻量化接口，快速返回补全文本
        
        Args:
            input_text: 输入文本
            model: 模型ID (如 o3-pro)
            stream: 是否流式输出
            background: 是否异步处理
            
        Returns:
            API 响应结果
        """
        url = f"{self.base_url}/responses"
        
        payload = {
            "model": model,
            "input": input_text,
            "stream": stream
        }
        
        if background:
            payload["background"] = background
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            self.logger.info(f"响应 API 调用成功，模型: {model}")
            return result
            
        except Exception as e:
            self.logger.error(f"响应 API 调用失败: {e}")
            raise EasyTransError(f"响应 API 调用失败: {e}")
    
    def messages(
        self,
        messages: List[Dict[str, str]],
        # model: str = "claude-opus-4-20250514",
        model: str = "claude-opus-4-1-20250805",
        max_tokens: int = 10000,
        stream: bool = False,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        消息 API - 构建复杂多轮对话，支持多模态输入
        
        Args:
            messages: 对话历史数组
            model: 模型ID (如 claude-opus-4-20250514)
            max_tokens: 最大生成令牌数
            stream: 是否流式输出
            system: 系统指令
            
        Returns:
            API 响应结果
        """
        url = f"{self.base_url}/messages"
        
        # 设置特殊头部
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            self.logger.info(f"消息 API 调用成功，模型: {model}")
            return result
            
        except Exception as e:
            self.logger.error(f"消息 API 调用失败: {e}")
            raise EasyTransError(f"消息 API 调用失败: {e}")
    
    def embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        encoding_format: str = "float"
    ) -> Dict[str, Any]:
        """
        嵌入 API - 将文本转换为向量表示
        
        Args:
            input_text: 文本或文本数组
            model: 嵌入模型ID
            encoding_format: 向量格式 (float 或 base64)
            
        Returns:
            API 响应结果
        """
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": model,
            "input": input_text,
            "encoding_format": encoding_format
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            self.logger.info(f"嵌入 API 调用成功，模型: {model}")
            return result
            
        except Exception as e:
            self.logger.error(f"嵌入 API 调用失败: {e}")
            raise EasyTransError(f"嵌入 API 调用失败: {e}")
    
    def get_response_result(self, response_id: str) -> Dict[str, Any]:
        """
        查询异步响应结果
        
        Args:
            response_id: 响应任务ID
            
        Returns:
            查询结果
        """
        url = f"{self.base_url}/responses/{response_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            result = response.json()
            
            self.logger.info(f"查询响应结果成功，ID: {response_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"查询响应结果失败: {e}")
            raise EasyTransError(f"查询响应结果失败: {e}")
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        验证 API 响应
        
        Args:
            response: API 响应
            
        Returns:
            验证是否通过
        """
        # 对话补全API响应格式
        if 'choices' in response:
            choice = response['choices'][0]
            return choice.get('finish_reason') in ['stop', None] or choice.get('message', {}).get('function_call')
        
        # 响应API格式
        if 'status' in response:
            return response['status'] in ['completed', 'queued', 'processing']
        
        # 消息API格式
        if 'role' in response and response['role'] == 'assistant':
            return 'content' in response
        
        # 嵌入API格式
        if 'data' in response and isinstance(response['data'], list):
            return len(response['data']) > 0
        
        return False
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        提取响应内容
        
        Args:
            response: API 响应
            
        Returns:
            提取的内容
        """
        # Messages API 响应格式 (Claude 模型)
        if 'content' in response and isinstance(response['content'], list):
            content_list = response['content']
            if content_list and isinstance(content_list[0], dict):
                text = content_list[0].get('text', '')
                if text:
                    return text
        
        # 对话补全API响应格式
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            
            # 检查是否因为长度限制被截断
            finish_reason = choice.get('finish_reason')
            if finish_reason == 'length':
                self.logger.warning("响应因长度限制被截断")
                # 尝试从其他字段获取内容或返回空字符串
                return ""
            
            message = choice.get('message', {})
            content = message.get('content', '')
            
            # 如果内容为空但有其他指示信息，记录日志
            if not content and finish_reason:
                self.logger.warning(f"响应内容为空，finish_reason: {finish_reason}")
            
            return content
        
        # 响应API格式
        if 'output' in response and isinstance(response['output'], list):
            for item in response['output']:
                if item.get('type') == 'message' and 'content' in item:
                    for content in item['content']:
                        if 'text' in content:
                            return content['text']
        
        # 消息API格式
        if 'content' in response and isinstance(response['content'], list):
            for content in response['content']:
                if content.get('type') == 'text':
                    return content.get('text', '')
        
        return ""
    
    def extract_function_call(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        提取函数调用（如果支持）
        
        Args:
            response: API 响应
            
        Returns:
            函数调用信息
        """
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            message = choice.get('message', {})
            return message.get('function_call')
        
        return None


class EasyTransError(Exception):
    """极易云 API 异常"""
    pass

if __name__ == "__main__":
    # 示例用法
    client = EasyTransClient()
    client.logger.info("极易云客户端初始化成功")

    # try:
    #     response = client.chat_completion(
    #         messages=[{"role": "user", "content": "你好，极易云！"}],
    #         model="gemini-2.5-pro"
    #     )
    #     print(response)
    #     content = client.extract_content(response)
    #     client.logger.info(f"响应内容: {content}")
    # except EasyTransError as e:
    #     client.logger.error(f"API 调用失败: {e}")

    try:
        response = client.responses(
            input_text="极易云开放平台是一个强大的AI服务平台",
            model="gpt-5-2025-08-07"
            # model="o3-pro-2025-06-10"
        )
        print(response)
        content = client.extract_content(response)
        client.logger.info(f"响应内容: {content}")          
    except EasyTransError as e:
        client.logger.error(f"响应 API 调用失败: {e}")

    # try:
    #     embedding_response = client.embeddings(
    #         input_text="极易云开放平台是一个强大的AI服务平台",
    #         model="text-embedding-3-small"
    #     )
    #     client.logger.info(f"嵌入 API 响应: {embedding_response}")
    # except EasyTransError as e:
    #     client.logger.error(f"嵌入 API 调用失败: {e}")
    # try:
    #     messages_response = client.messages(
    #         messages=[{"role": "user", "content": "请帮我计算f980共振态"}]
    #     )
    #     content = client.extract_content(messages_response)
    #     client.logger.info(f"消息 API 响应内容: {content}")
    # except EasyTransError as e:
    #     client.logger.error(f"消息 API 调用失败: {e}")
