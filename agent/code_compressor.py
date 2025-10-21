#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
代码压缩工具 - 用于压缩生成的Python代码
可以被llm_code_generator.py调用，以减少代码体积并优化LLM处理效率
"""

import re
import ast
import tokenize
import io
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import logging

# 尝试导入easytrans_client，如果可用则使用LLM进行智能压缩
try:
    from agent.easytrans_client import EasyTransClient
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False

logger = logging.getLogger(__name__)


class CodeCompressor:
    """代码压缩工具类，提供多种压缩级别和方法"""
    
    def __init__(self, llm_client=None, model: str = "o3-pro-2025-06-10"):
        """
        初始化代码压缩器
        
        Args:
            llm_client: 可选的LLM客户端实例，用于智能压缩
            model: 使用的LLM模型名称
        """
        self.llm_client = llm_client
        self.model = model
        
        # 变量名映射字典
        self.var_mapping = {}
        # 已使用的短变量名集合
        self.used_names = set()
        # 保留的变量名（不会被重命名）
        self.preserved_names = {
            'self', 'cls', 'args', 'kwargs', 'np', 'onp', 'jnp', 'jax', 
            'plt', 'pd', 'os', 'sys', 'time', 'logging', 'config', 'data'
        }
        
    def compress_code(self, code: str, level: str = 'medium', 
                     use_llm: bool = False) -> str:
        """
        压缩Python代码
        
        Args:
            code: 要压缩的Python代码字符串
            level: 压缩级别，可选值为'light', 'medium', 'heavy'
            use_llm: 是否使用LLM进行智能压缩
            
        Returns:
            压缩后的代码字符串
        """
        if not code.strip():
            return code
            
        # 基本压缩（所有级别都会执行）
        compressed = self._remove_comments(code)
        compressed = self._remove_extra_whitespace(compressed)
        
        # 根据压缩级别选择不同的压缩策略
        if level in ('medium', 'heavy'):
            compressed = self._optimize_imports(compressed)
            compressed = self._shorten_docstrings(compressed)
            
        if level == 'heavy':
            # 重命名变量（最激进的压缩方式）
            compressed = self._rename_variables(compressed)
        
        # 使用LLM进行智能压缩（如果启用）
        if use_llm and HAS_LLM_CLIENT and self.llm_client:
            compressed = self._llm_compress(compressed)
            
        return compressed
    
    def _remove_comments(self, code: str) -> str:
        """移除代码中的注释，但保留文档字符串"""
        result = []
        in_string = False
        string_char = None
        i = 0
        
        while i < len(code):
            # 处理字符串
            if code[i] in ('"', "'") and (i == 0 or code[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = code[i]
                elif code[i] == string_char:
                    in_string = False
                result.append(code[i])
                
            # 处理注释
            elif code[i] == '#' and not in_string:
                # 跳过这一行的剩余部分
                line_end = code.find('\n', i)
                if line_end == -1:  # 如果是最后一行
                    break
                i = line_end
                result.append('\n')
            else:
                result.append(code[i])
            
            i += 1
            
        return ''.join(result)
    
    def _remove_extra_whitespace(self, code: str) -> str:
        """移除多余的空白行和行尾空格"""
        # 移除行尾空格
        lines = [line.rstrip() for line in code.splitlines()]
        
        # 移除多余的空行（保留单个空行用于分隔代码块）
        compressed_lines = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                compressed_lines.append(line)
            prev_empty = is_empty
            
        return '\n'.join(compressed_lines)
    
    def _optimize_imports(self, code: str) -> str:
        """优化导入语句，合并相同模块的导入"""
        try:
            tree = ast.parse(code)
            imports = {}
            
            # 收集所有导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        alias = name.asname if name.asname else name.name
                        imports.setdefault(name.name, []).append(alias)
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        alias = name.asname if name.asname else name.name
                        key = f"from {module} import"
                        imports.setdefault(key, []).append((name.name, alias))
            
            # 重建导入语句
            import_lines = []
            for module, aliases in imports.items():
                if module.startswith("from "):
                    names = []
                    for name, alias in aliases:
                        if name == alias:
                            names.append(name)
                        else:
                            names.append(f"{name} as {alias}")
                    import_lines.append(f"{module} {', '.join(names)}")
                else:
                    for alias in aliases:
                        if module == alias:
                            import_lines.append(f"import {module}")
                        else:
                            import_lines.append(f"import {module} as {alias}")
            
            # 替换原始导入语句
            import_block = '\n'.join(import_lines)
            
            # 找到所有导入语句的范围
            import_nodes = [n for n in ast.walk(tree) 
                          if isinstance(n, (ast.Import, ast.ImportFrom))]
            
            if not import_nodes:
                return code
                
            min_lineno = min(node.lineno for node in import_nodes)
            max_lineno = max(node.end_lineno if hasattr(node, 'end_lineno') else node.lineno 
                           for node in import_nodes)
            
            # 分割代码
            lines = code.splitlines()
            before = '\n'.join(lines[:min_lineno-1])
            after = '\n'.join(lines[max_lineno:])
            
            # 重组代码
            return f"{before}\n{import_block}\n{after}"
            
        except SyntaxError:
            # 如果代码有语法错误，返回原始代码
            logger.warning("无法解析代码AST，跳过导入优化")
            return code
    
    def _shorten_docstrings(self, code: str) -> str:
        """缩短文档字符串，保留第一行"""
        def replace_docstring(match):
            docstring = match.group(1)
            lines = docstring.strip().split('\n')
            if len(lines) <= 1:
                return match.group(0)  # 如果只有一行，保持不变
                
            # 保留第一行，去掉其余部分
            first_line = lines[0].strip()
            return f'"""{first_line}"""'
            
        # 匹配三引号文档字符串
        pattern = r'"""(.*?)"""'
        return re.sub(pattern, replace_docstring, code, flags=re.DOTALL)
    
    def _rename_variables(self, code: str) -> str:
        """重命名变量为更短的名称"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.warning("无法解析代码AST，跳过变量重命名")
            return code
            
        # 收集所有变量名
        variables = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in self.preserved_names:
                    variables.add(node.id)
        
        # 生成短变量名映射
        self.var_mapping = {}
        for var in variables:
            short_name = self._generate_short_name()
            self.var_mapping[var] = short_name
        
        # 替换变量名
        for var, short_name in self.var_mapping.items():
            # 使用正则表达式替换完整的变量名（避免部分匹配）
            pattern = r'\b{}\b'.format(re.escape(var))
            code = re.sub(pattern, short_name, code)
            
        return code
    
    def _generate_short_name(self) -> str:
        """生成短变量名，确保不包含特殊字符且不是Python关键字"""
        # Python关键字列表
        python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 
            'with', 'yield'
        }
        
        # 只使用字母和数字，避免特殊字符
        # 单字母变量名
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c not in self.used_names and c not in self.preserved_names and c not in python_keywords:
                self.used_names.add(c)
                return c
                
        # 双字母变量名
        for i in 'abcdefghijklmnopqrstuvwxyz':
            for j in 'abcdefghijklmnopqrstuvwxyz':
                name = i + j
                if name not in self.used_names and name not in self.preserved_names and name not in python_keywords:
                    self.used_names.add(name)
                    return name
        
        # 如果上面都用完了，使用字母+数字的组合
        for c in 'abcdefghijklmnopqrstuvwxyz':
            for i in range(10):  # 使用数字0-9
                name = f"{c}{i}"
                if name not in self.used_names and name not in self.preserved_names and name not in python_keywords:
                    self.used_names.add(name)
                    return name
        
        # 最后的备选方案
        i = 0
        while True:
            name = f"v{i}"
            if name not in self.used_names and name not in self.preserved_names and name not in python_keywords:
                self.used_names.add(name)
                return name
            i += 1
    
    def _llm_compress(self, code: str) -> str:
        """使用LLM进行智能代码压缩"""
        if not self.llm_client:
            logger.warning("LLM客户端未初始化，跳过LLM压缩")
            return code
            
        prompt = f"""
请压缩以下Python代码，使其更简洁但保持完全相同的功能。
你的目标是：
1. 移除所有注释和不必要的文档字符串
2. 缩短变量名和函数名（但保持可读性）
3. 合并可以合并的语句
4. 减少空行和空格
5. 使用更简洁的Python语法

重要：代码必须保持完全相同的功能和行为！

代码：
```python
{code}
```

只返回压缩后的代码，不要有任何解释或额外文本。
"""
        
        try:
            response = self.llm_client.responses(
                input_text=prompt,
                model=self.model
            )
            
            compressed_code = self.llm_client.extract_content(response)
            
            # 验证压缩后的代码是否有效
            try:
                ast.parse(compressed_code)
                return compressed_code
            except SyntaxError:
                logger.warning("LLM生成的代码存在语法错误，返回原始压缩代码")
                return code
                
        except Exception as e:
            logger.error(f"LLM压缩失败: {e}")
            return code


def compress_code(code: str, level: str = 'medium', use_llm: bool = False,
                 llm_client=None, model: str = "o3-pro-2025-06-10") -> str:
    """
    压缩Python代码的便捷函数
    
    Args:
        code: 要压缩的Python代码字符串
        level: 压缩级别，可选值为'light', 'medium', 'heavy'
        use_llm: 是否使用LLM进行智能压缩
        llm_client: 可选的LLM客户端实例
        model: 使用的LLM模型名称
        
    Returns:
        压缩后的代码字符串
    """
    original_size = len(code)
    print(f"🔄 正在压缩代码... (原始大小: {original_size} 字符)")
    try:
        compressor = CodeCompressor(llm_client=llm_client, model=model)
        compressed_code = compressor.compress_code(code, level=level, use_llm=use_llm)
        compressed_size = len(compressed_code)
        print(f"✅ 代码压缩完成! (压缩大小: {compressed_size} 字符)")
        return compressed_code
    except Exception as e:
        logger.error(f"代码压缩失败: {e}")
        return code


if __name__ == "__main__":
    # 简单的命令行接口用于测试
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Python代码压缩工具")
    parser.add_argument("file", nargs="?", help="要压缩的Python文件")
    parser.add_argument("--level", choices=["light", "medium", "heavy"], 
                      default="medium", help="压缩级别")
    parser.add_argument("--use-llm", action="store_true", help="使用LLM进行智能压缩")
    parser.add_argument("--output", "-o", help="输出文件，默认为标准输出")
    
    args = parser.parse_args()
    
    # 读取输入
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            code = f.read()
    else:
        code = sys.stdin.read()
    
    # 初始化LLM客户端（如果需要）
    llm_client = None
    if args.use_llm and HAS_LLM_CLIENT:
        try:
            from agent.easytrans_client import EasyTransClient
            llm_client = EasyTransClient()
        except Exception as e:
            print(f"无法初始化LLM客户端: {e}", file=sys.stderr)
    
    # 压缩代码
    compressed = compress_code(code, level=args.level, use_llm=args.use_llm,
                             llm_client=llm_client)
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(compressed)
    else:
        print(compressed)