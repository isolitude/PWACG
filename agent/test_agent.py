#!/usr/bin/env python3
# coding: utf-8
"""
Agent 功能测试脚本
"""

import os
import json
import tempfile
from .code_generator import PWACodeGenerator
from .openai_client import OpenAIClient


def test_openai_client():
    """测试 OpenAI 客户端"""
    print("=== 测试 OpenAI 客户端 ===")
    
    try:
        # 使用环境变量或测试配置
        client = OpenAIClient()
        
        # 简单的聊天测试
        messages = [
            {"role": "user", "content": "请生成一个简单的 Python hello world 函数"}
        ]
        
        response = client.chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=200
        )
        
        print("✓ OpenAI 客户端工作正常")
        print(f"响应内容: {client.extract_content(response)[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI 客户端测试失败: {e}")
        return False


def test_code_generator():
    """测试代码生成器"""
    print("=== 测试代码生成器 ===")
    
    try:
        # 配置
        config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'model': 'gpt-3.5-turbo'
        }
        
        generator = PWACodeGenerator(config)
        
        # 测试数据
        template_data = {
            "module_name": "test_fit",
            "data_path": "/path/to/data",
            "output_path": "/path/to/output",
            "parameters": {
                "max_iterations": 1000,
                "tolerance": 1e-6
            }
        }
        
        # 生成测试代码
        generated_code = generator.generate_fit_code(template_data)
        
        # 验证代码
        if generator.validate_output(generated_code):
            print("✓ 代码生成和验证成功")
            print("生成的代码片段:")
            print(generated_code[:300] + "...")
            return True
        else:
            print("✗ 代码验证失败")
            return False
            
    except Exception as e:
        print(f"✗ 代码生成器测试失败: {e}")
        return False


def test_function_calling():
    """测试 Function Calling 功能"""
    print("=== 测试 Function Calling ===")
    
    try:
        config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'model': 'gpt-3.5-turbo'
        }
        
        client = OpenAIClient()
        
        # 定义测试函数
        functions = [
            {
                "name": "generate_test_code",
                "description": "生成测试代码",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "生成的代码"
                        },
                        "language": {
                            "type": "string",
                            "description": "编程语言"
                        }
                    },
                    "required": ["code", "language"]
                }
            }
        ]
        
        messages = [
            {
                "role": "user", 
                "content": "请使用 generate_test_code 函数生成一个简单的 Python 加法函数"
            }
        ]
        
        response = client.chat_completion(
            messages=messages,
            functions=functions,
            function_call={"name": "generate_test_code"}
        )
        
        function_call = client.extract_function_call(response)
        if function_call and function_call.get('name') == 'generate_test_code':
            arguments = json.loads(function_call['arguments'])
            print("✓ Function Calling 工作正常")
            print(f"生成的代码: {arguments.get('code', '')[:100]}...")
            print(f"语言: {arguments.get('language', '')}")
            return True
        else:
            print("✗ Function Calling 未正确工作")
            return False
            
    except Exception as e:
        print(f"✗ Function Calling 测试失败: {e}")
        return False


def test_file_operations():
    """测试文件操作"""
    print("=== 测试文件操作 ===")
    
    try:
        config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY', 'test_key'),
            'model': 'gpt-3.5-turbo'
        }
        
        generator = PWACodeGenerator(config)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_code = """
def hello_world():
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
"""
            temp_file = f.name
        
        # 测试保存功能
        success = generator.save_output(test_code, temp_file)
        
        if success and os.path.exists(temp_file):
            print("✓ 文件操作测试成功")
            # 清理临时文件
            os.unlink(temp_file)
            return True
        else:
            print("✗ 文件操作测试失败")
            return False
            
    except Exception as e:
        print(f"✗ 文件操作测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("开始运行 Agent 功能测试...\n")
    
    tests = [
        ("OpenAI 客户端测试", test_openai_client),
        ("Function Calling 测试", test_function_calling),
        ("文件操作测试", test_file_operations),
        ("代码生成器测试", test_code_generator),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 执行异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*50}")
    print("测试结果汇总:")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
    else:
        print("⚠️  部分测试失败，请检查配置和环境")
    
    return passed == total


if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv('OPENAI_API_KEY'):
        print("警告: OPENAI_API_KEY 环境变量未设置，某些测试可能会失败")
        print("请设置环境变量: export OPENAI_API_KEY='your-api-key'")
    
    run_all_tests()