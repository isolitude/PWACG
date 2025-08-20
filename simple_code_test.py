#!/usr/bin/env python3
# coding: utf-8
"""
简单的EasyTransNote API代码生成测试脚本
快速验证API连接和代码生成功能
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.code_generator import PWACodeGenerator


def save_generated_code(code, code_type, test_number):
    """保存生成的代码到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_code_{code_type}_test{test_number}_{timestamp}.py"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f'"""\n生成的{code_type}代码 - 测试{test_number}\n')
            f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'代码长度: {len(code)} 字符\n"""\n\n')
            f.write(code)
        
        print(f"✓ 代码已保存到: {filename}")
        return filename
        
    except Exception as e:
        print(f"✗ 保存代码失败: {e}")
        return None


def test_simple_code_generation():
    """测试基本代码生成功能"""
    print("=== EasyTransNote API 代码生成测试 ===\n")
    
    # API配置
    config = {
        'api_provider': 'easytrans',
        'api_key': os.getenv('EASYTRANS_API_KEY', 'sk-v1-LJfF-3OKN3ZyESALnL08vWbfSOQ-MYIop/n/'),
        'model': 'gemini-2.5-pro',  # 推荐的代码生成模型
        'base_url': 'https://api.easytransnote.com/v1'
    }
    
    print(f"使用模型: {config['model']}")
    print(f"API提供商: {config['api_provider']}")
    print("-" * 50)
    
    try:
        # 创建代码生成器
        generator = PWACodeGenerator(config)
        
        # 测试数据1: 简单的数据处理函数
        test_data_1 = {
            "task": "创建一个数据处理函数",
            "description": "读取CSV文件，进行数据清洗和基本统计分析",
            "input_file": "data.csv",
            "output_file": "processed_data.csv"
        }
        
        print("测试1: 生成数据处理代码...")
        code_1 = generator.generate_code(test_data_1, "data_processing")
        print("✓ 生成成功")
        print(f"代码长度: {len(code_1)} 字符")
        
        # 保存生成的代码
        save_generated_code(code_1, "data_processing", 1)
        
        print("生成的代码片段:")
        print("-" * 30)
        print(code_1[:300] + "...\n" if len(code_1) > 300 else code_1 + "\n")
        
        # 测试数据2: 绘图函数
        test_data_2 = {
            "task": "创建一个数据可视化函数", 
            "description": "读取数据文件并生成散点图和直方图",
            "plot_types": ["scatter", "histogram"],
            "save_format": "png"
        }
        
        print("测试2: 生成数据可视化代码...")
        code_2 = generator.generate_code(test_data_2, "visualization")
        print("✓ 生成成功")
        print(f"代码长度: {len(code_2)} 字符")
        
        # 保存生成的代码
        save_generated_code(code_2, "visualization", 2)
        
        print("生成的代码片段:")
        print("-" * 30)
        print(code_2[:300] + "...\n" if len(code_2) > 300 else code_2 + "\n")
        
        # 测试数据3: 数学计算函数
        test_data_3 = {
            "task": "创建数学计算函数",
            "description": "实现矩阵运算和数值积分功能",
            "functions": ["matrix_multiply", "numerical_integration"],
            "libraries": ["numpy", "scipy"]
        }
        
        print("测试3: 生成数学计算代码...")
        code_3 = generator.generate_code(test_data_3, "math_computation")
        print("✓ 生成成功")
        print(f"代码长度: {len(code_3)} 字符")
        
        # 保存生成的代码
        save_generated_code(code_3, "math_computation", 3)
        
        print("生成的代码片段:")
        print("-" * 30)
        print(code_3[:300] + "...\n" if len(code_3) > 300 else code_3 + "\n")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_different_models():
    """测试不同模型的代码生成"""
    print("=== 不同模型对比测试 ===\n")
    
    models = [
        'gemini-2.5-pro',
        'gpt-4o', 
        'claude-opus-4-20250514'
    ]
    
    test_data = {
        "task": "创建一个简单的Web API",
        "description": "使用Flask创建一个REST API，包含GET和POST端点",
        "framework": "Flask"
    }
    
    for model in models:
        print(f"测试模型: {model}")
        config = {
            'api_provider': 'easytrans',
            'api_key': os.getenv('EASYTRANS_API_KEY', 'sk-v1-4AZX-WT7NBe+1tpuzCVaV5ObJ7w-/obBkWCB'),
            'model': model
        }
        
        try:
            generator = PWACodeGenerator(config)
            code = generator.generate_code(test_data, "web_api")
            print(f"✓ {model} 生成成功 ({len(code)} 字符)")
            
        except Exception as e:
            print(f"✗ {model} 生成失败: {e}")
        
        print("-" * 30)


def test_code_validation():
    """测试代码验证功能"""
    print("=== 代码验证测试 ===\n")
    
    config = {
        'api_provider': 'easytrans',
        'api_key': os.getenv('EASYTRANS_API_KEY', 'sk-v1-4AZX-WT7NBe+1tpuzCVaV5ObJ7w-/obBkWCB'),
        'model': 'gemini-2.5-pro'
    }
    
    try:
        generator = PWACodeGenerator(config)
        
        test_data = {
            "task": "创建一个类",
            "description": "定义一个学生信息管理类，包含基本属性和方法",
            "class_name": "Student",
            "methods": ["add_student", "get_student", "update_grade"]
        }
        
        print("生成代码...")
        code = generator.generate_code(test_data, "class_definition")
        
        print("验证生成的代码...")
        if generator.validate_output(code):
            print("✓ 代码验证通过")
            print("✓ 语法正确，结构完整")
            
            # 保存到文件进行进一步检查
            filename = save_generated_code(code, "class_definition", "validation")
            if filename:
                print(f"✓ 验证通过的代码已保存")
            
        else:
            print("✗ 代码验证失败")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def main():
    """主函数"""
    print("EasyTransNote API 代码生成测试工具")
    print("=" * 60)
    
    # 检查API密钥
    api_key = os.getenv('EASYTRANS_API_KEY')
    if not api_key:
        print("⚠️  未检测到 EASYTRANS_API_KEY 环境变量")
        print("使用内置测试密钥进行测试...")
    else:
        print(f"✓ 检测到API密钥: {api_key[:10]}...")
    
    print("\n")
    
    # 运行测试
    tests = [
        ("基本代码生成测试", test_simple_code_generation),
        ("不同模型对比测试", test_different_models), 
        ("代码验证测试", test_code_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试执行出错: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<25} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！EasyTransNote API工作正常")
    else:
        print("⚠️  部分测试失败，请检查API配置或网络连接")


if __name__ == "__main__":
    main()