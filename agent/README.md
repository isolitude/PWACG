# LLM Agent 代码生成器

PWACG 的智能代码生成系统，支持 OpenAI 和极易云开放平台 API，用于替代传统 Jinja2 模板的现代化代码生成解决方案。

## 🚀 功能特性

- ✅ **双 API 支持**：支持 OpenAI API 和极易云开放平台 API
- ✅ **智能 API 路由**：根据模型自动选择最适合的 API 接口
- ✅ **多模型支持**：GPT、Gemini、Claude、O3 等主流模型
- ✅ **Function Calling**：严格控制输出格式（OpenAI）
- ✅ **多种代码类型**：fit、draw、tensor 等专业代码生成
- ✅ **批量生成**：一次性生成所有必需脚本
- ✅ **智能验证**：AST 语法检查和代码质量验证
- ✅ **故障转移**：API 失败时自动切换备用方案

## 📁 文件结构

```
agent/
├── __init__.py                 # 包初始化
├── base.py                     # 基础抽象类
├── openai_client.py            # OpenAI API 客户端
├── easytrans_client.py         # 极易云 API 客户端（新增）
├── code_generator.py           # 智能代码生成器
├── llm_create_control.py       # 完整工作流程控制器
├── test_agent.py              # OpenAI 测试脚本
├── easytrans_example.py       # 极易云测试和示例脚本
└── README.md                  # 本文档
```

## ⚙️ 环境设置

### 1. 安装依赖

```bash
# 基础依赖
pip install openai requests

# 或者从项目根目录安装
pip install -r requirements.txt
```

### 2. API 密钥配置

#### 选项 A：OpenAI API（默认）
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
```

#### 选项 B：极易云开放平台 API（推荐）
```bash
export EASYTRANS_API_KEY="your-easytrans-api-key"
export EASYTRANS_BASE_URL="https://api.easytransnote.com/v1"  # 可选
```

## 📖 使用指南

### 🔧 快速开始

#### 1. 基础代码生成

```python
from agent.code_generator import PWACodeGenerator

# 基础配置
config = {
    'api_provider': 'easytrans',  # 或 'openai'
    'api_key': 'your-api-key',
    'model': 'gemini-2.5-pro'     # 推荐用于代码生成
}

# 创建生成器
generator = PWACodeGenerator(config)

# 准备模板数据
template_data = {
    "module_name": "fit_analysis",
    "data_path": "/path/to/data",
    "parameters": {"max_iterations": 1000, "tolerance": 1e-6}
}

# 生成不同类型的代码
fit_code = generator.generate_fit_code(template_data)        # 拟合代码
draw_code = generator.generate_draw_code(template_data)      # 绘图代码
tensor_code = generator.generate_tensor_code(template_data) # 张量计算代码

print("生成的拟合代码：")
print(fit_code[:200] + "...")
```

#### 2. 完整工作流程（替代 Jinja2）

```python
from agent.llm_create_control import LLMCreateCode
import json

# 加载项目配置
with open("config/generator_kk.json", encoding='utf-8') as f:
    dict_json = json.loads(f.read())

# LLM Agent 配置
agent_config = {
    'api_provider': 'easytrans',
    'api_key': 'your-easytrans-api-key',
    'model': 'gemini-2.5-pro'
}

# 创建 LLM 控制器
llm_creator = LLMCreateCode(dict_json, agent_config)

# 一键生成所有代码（完全替代传统 create_all_scripts.py）
llm_creator.generate_all()

# 或者分步生成
# llm_creator.llm_generate_fit()     # 只生成拟合代码
# llm_creator.llm_generate_draw()    # 只生成绘图代码
# llm_creator.llm_generate_tensor()  # 只生成张量代码
```

### 🎯 支持的模型和 API

系统会根据模型名称自动选择最适合的 API：

| 模型系列 | 使用的 API | 推荐用途 | 示例模型 |
|---------|-----------|---------|----------|
| **Gemini** | Chat Completions | 代码生成 | `gemini-2.5-pro` |
| **GPT** | Chat Completions | 通用任务 | `gpt-4o` |
| **Claude** | Messages | 复杂推理 | `claude-opus-4-20250514`, `claude-sonnet-4-20250514` |
| **O3** | Responses | 高级推理 | `o3-pro-2025-06-10` |

#### 极易云模型配置示例

```python
# Gemini 模型（推荐用于代码生成）
config = {
    'api_provider': 'easytrans',
    'api_key': 'your-easytrans-api-key',
    'model': 'gemini-2.5-pro'
}

# Claude Opus（适合复杂代码生成）
config = {
    'api_provider': 'easytrans',
    'api_key': 'your-easytrans-api-key',
    'model': 'claude-opus-4-20250514'
}

# O3 Pro（最新推理模型）
config = {
    'api_provider': 'easytrans',
    'api_key': 'your-easytrans-api-key',
    'model': 'o3-pro-2025-06-10'
}
```

### 🔄 从传统模板迁移

#### 原有方式（Jinja2）
```bash
# 传统方式
python create_all_scripts.py
```

#### 新方式（LLM Agent）
```bash
# 使用 LLM Agent
python agent/llm_create_control.py

# 或者在代码中
from agent.llm_create_control import LLMCreateCode
# ... 如上所示的完整工作流程代码
```

### 🧪 测试和验证

#### 测试 OpenAI API
```bash
cd agent
export OPENAI_API_KEY="your-openai-api-key"
python test_agent.py
```

#### 测试极易云 API
```bash
cd agent
export EASYTRANS_API_KEY="your-easytrans-api-key"
python easytrans_example.py
```

#### 测试输出示例
```
=== 极易云不同模型测试 ===

测试模型: gemini-2.5-pro (对话补全API)
✓ gemini-2.5-pro 测试成功
生成代码长度: 1254 字符
使用的API: 对话补全API

测试模型: claude-opus-4-20250514 (消息API)
✓ claude-opus-4-20250514 测试成功
生成代码长度: 1456 字符
使用的API: 消息API
```

## 🛠️ 高级用法

### 批量代码生成

```python
from agent.code_generator import PWACodeGenerator

config = {
    'api_provider': 'easytrans',
    'api_key': 'your-easytrans-api-key',
    'model': 'gemini-2.5-pro'
}

generator = PWACodeGenerator(config)

# 批量配置
batch_configs = [
    {
        "type": "fit",
        "data": {
            "module_name": "kk_analysis",
            "data_path": "data/kk/",
            "parameters": {"max_iterations": 1000}
        }
    },
    {
        "type": "draw", 
        "data": {
            "module_name": "kk_plotting",
            "output_path": "output/pictures/"
        }
    },
    {
        "type": "tensor",
        "data": {
            "module_name": "tensor_calc",
            "cache_config": {"enable": True}
        }
    }
]

# 批量生成
results = generator.batch_generate(batch_configs, output_dir="rendered_scripts")

# 检查结果
for i, success in enumerate(results):
    status = "✓ 成功" if success else "✗ 失败"
    print(f"任务 {i+1}: {status}")
```

### 自定义提示和验证

```python
from agent.code_generator import CodeGenerator

class CustomCodeGenerator(CodeGenerator):
    def _build_prompt(self, template_data, code_type):
        """自定义提示构建"""
        base_prompt = super()._build_prompt(template_data, code_type)
        
        # 添加自定义要求
        custom_requirements = """
额外要求：
- 添加详细的 docstring
- 使用类型提示
- 包含错误处理
- 遵循 PEP 8 规范
"""
        return base_prompt + custom_requirements
    
    def validate_output(self, generated_code):
        """自定义验证"""
        # 基础验证
        if not super().validate_output(generated_code):
            return False
        
        # 自定义验证规则
        if 'def ' not in generated_code:
            self.logger.warning("代码中缺少函数定义")
            return False
            
        if '"""' not in generated_code:
            self.logger.warning("代码中缺少 docstring")
            
        return True

# 使用自定义生成器
custom_generator = CustomCodeGenerator(config)
code = custom_generator.generate_code(template_data, "fit")
```

## 🔍 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   ValueError: API 密钥未设置，请设置 EASYTRANS_API_KEY 环境变量
   ```
   **解决**：检查环境变量设置
   ```bash
   echo $EASYTRANS_API_KEY  # 检查是否设置
   export EASYTRANS_API_KEY="your-api-key"
   ```

2. **模型不支持**
   ```
   CodeGenerationError: 极易云 API 调用失败
   ```
   **解决**：检查模型名称是否正确，或尝试备用模型

3. **生成代码语法错误**
   ```
   ValidationError: 代码语法错误: invalid syntax
   ```
   **解决**：检查模板数据格式，或尝试不同的模型

### 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

config = {
    'api_provider': 'easytrans',
    'api_key': 'your-api-key',
    'model': 'gemini-2.5-pro'
}

generator = PWACodeGenerator(config)
# 现在会看到详细的调试信息
```

## 📊 性能对比

| 特性 | 传统 Jinja2 | LLM Agent | 备注 |
|------|-------------|-----------|------|
| **设置时间** | 快 | 中等 | 需要 API 配置 |
| **生成速度** | 极快 | 中等 | 依赖网络和模型 |
| **代码质量** | 固定 | 动态优化 | AI 可以改进代码 |
| **灵活性** | 低 | 极高 | 可适应不同需求 |
| **维护成本** | 高 | 低 | 无需维护模板 |
| **错误处理** | 基础 | 智能 | 自动修复常见问题 |

## 🎯 最佳实践

### 1. 模型选择建议

- **代码生成**：`gemini-2.5-pro`（速度快，质量好）
- **复杂逻辑**：`claude-opus-4-20250514`（推理能力强）
- **高质量输出**：`o3-pro-2025-06-10`（最新模型）
- **成本敏感**：`gpt-4o`（性价比高）

### 2. 配置管理

```python
# config/agent_config.json
{
  "production": {
    "api_provider": "easytrans",
    "model": "gemini-2.5-pro",
    "temperature": 0.1
  },
  "development": {
    "api_provider": "easytrans", 
    "model": "gpt-4o",
    "temperature": 0.3
  }
}

# 在代码中使用
import json
import os

env = os.getenv('ENVIRONMENT', 'development')
with open('config/agent_config.json') as f:
    config = json.load(f)[env]
    config['api_key'] = os.getenv('EASYTRANS_API_KEY')
```

### 3. 错误恢复

```python
def robust_generate(template_data, code_type):
    """带故障恢复的代码生成"""
    models = ['gemini-2.5-pro', 'gpt-4o', 'claude-opus-4-20250514']
    
    for model in models:
        try:
            config = {
                'api_provider': 'easytrans',
                'api_key': os.getenv('EASYTRANS_API_KEY'),
                'model': model
            }
            
            generator = PWACodeGenerator(config)
            return generator.generate_code(template_data, code_type)
            
        except Exception as e:
            print(f"模型 {model} 失败: {e}")
            continue
    
    raise Exception("所有模型都失败了")
```

## 🔗 相关链接

- [极易云开放平台文档](https://docs.easytransnote.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [PWACG 项目主页](https://github.com/your-repo/PWACG)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 添加测试用例
4. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件。