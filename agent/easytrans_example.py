#!/usr/bin/env python3
# coding: utf-8
"""
极易云 API 使用示例
演示如何使用极易云开放平台进行代码生成
"""

import os
import sys
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.code_generator import PWACodeGenerator
from agent.llm_create_control import LLMCreateCode


def test_easytrans_basic():
    """测试极易云基本功能"""
    print("=== 极易云基本功能测试 ===")
    
    # 极易云配置
    config = {
        'api_provider': 'easytrans',  # 指定使用极易云
        'model': 'gemini-2.5-pro-stable',  # 使用Gemini模型
        'base_url': 'https://api.easytransnote.com/v1'  # 可选，默认就是这个
    }
    
    try:
        generator = PWACodeGenerator(config)
        
        # 测试数据
        template_data = {
            "module_name": "test_fit",
            "data_path": "/home/sean/workspace/PWACG/data",
            "parameters": {"max_iterations": 1000}
        }
        
        # 生成拟合代码
        print("正在生成拟合代码...")
        fit_code = generator.generate_fit_code(template_data)
        print("生成成功！代码片段:")
        print(fit_code[:300] + "...")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


def test_easytrans_full_workflow():
    """测试完整的极易云工作流程"""
    print("\n=== 极易云完整工作流程测试 ===")
    
    # 检查配置文件
    config_file = "config/generator_kk.json"
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在")
        return False
    
    with open(config_file, encoding='utf-8') as f:
        dict_json = json.loads(f.read())
    
    # 极易云Agent配置
    agent_config = {
        'api_provider': 'easytrans',
        'model': 'gemini-2.5-pro'
    }
    
    try:
        # 创建LLM代码生成器
        llm_creator = LLMCreateCode(dict_json, agent_config)
        
        # 生成拟合代码
        print("正在生成拟合代码...")
        llm_creator.llm_generate_fit()
        
        # 生成绘图代码  
        print("正在生成绘图代码...")
        llm_creator.llm_generate_draw()
        
        # 生成张量计算代码
        print("正在生成张量计算代码...")
        llm_creator.llm_generate_tensor()
        
        print("完整工作流程测试成功！")
        return True
        
    except Exception as e:
        print(f"完整工作流程测试失败: {e}")
        return False


def test_different_models():
    """测试不同的极易云模型"""
    print("\n=== 极易云不同模型测试 ===")
    
    models = [
        ('gemini-2.5-pro', '对话补全API'),
        ('gpt-4o', '对话补全API'), 
        ('o3-pro-2025-06-10', '响应API'),
        ('claude-opus-4-20250514', '消息API'),
        ('claude-sonnet-4-20250514', '消息API')
    ]
    
    template_data = {
        "module_name": "test_simple",
        "task": "创建一个简单的数据处理函数"
    }
    
    for model, api_type in models:
        print(f"\n测试模型: {model} ({api_type})")
        config = {
            'api_provider': 'easytrans',
            'api_key': 'sk-v1-4AZX-WT7NBe+1tpuzCVaV5ObJ7w-/obBkWCB',
            'model': model
        }
        
        try:
            generator = PWACodeGenerator(config)
            code = generator.generate_code(template_data, "generic")
            print(f"✓ {model} 测试成功")
            print(f"生成代码长度: {len(code)} 字符")
            print(f"使用的API: {api_type}")
            
        except Exception as e:
            print(f"✗ {model} 测试失败: {e}")


def main():
    """主函数"""
    print("极易云开放平台 Agent 使用示例")
    print("=" * 50)
    
    # 检查API密钥（已在代码中硬编码，跳过环境变量检查）
    # if not os.getenv('EASYTRANS_API_KEY'):
    #     print("警告: EASYTRANS_API_KEY 环境变量未设置")
    #     print("请设置: export EASYTRANS_API_KEY='your-api-key'")
    #     print("\n显示配置示例:")
    #     show_usage_examples()
    #     return
    
    # 运行测试
    results = []
    
    # 基本功能测试
    # results.append(("基本功能测试", test_easytrans_basic()))
    
    # # 不同模型测试
    # results.append(("不同模型测试", test_different_models()))
    
    # 完整工作流程测试（需要配置文件）
    if os.path.exists("config/generator_kk.json"):
        results.append(("完整工作流程测试", test_easytrans_full_workflow()))
    else:
        print("\n跳过完整工作流程测试（缺少配置文件）")
    
    # 显示结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
    

if __name__ == "__main__":
    main()