# LLM Agent 代码生成器

这个 agent 系统用于替代项目中的 Jinja2 模板功能，使用 LLM 和 Function Calling 来智能生成代码。

## 功能特性

- ✅ 使用 OpenAI API 进行智能代码生成
- ✅ Function Calling 严格控制输出格式
- ✅ 支持多种代码类型生成（fit、draw、tensor等）
- ✅ 代码语法验证和错误检查
- ✅ 批量代码生成能力
- ✅ 完整的日志记录系统

## 文件结构

```
agent/
├── __init__.py              # 包初始化文件
├── base.py                  # 基础抽象类
├── openai_client.py         # OpenAI API 客户端
├── code_generator.py        # 核心代码生成器
├── llm_create_control.py    # LLM 版本的创建控制器
├── test_agent.py           # 功能测试脚本
└── README.md               # 本文档
```

## 环境设置

### 1. 安装依赖

```bash
pip install openai
```

### 2. 设置环境变量

```bash
# OpenAI API 密钥（必需）
export OPENAI_API_KEY="your-openai-api-key"

# 自定义 API 端点（可选）
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 使用方法

### 基本使用

```python
from agent.code_generator import PWACodeGenerator

# 配置
config = {
    'openai_api_key': 'your-api-key',
    'model': 'gpt-3.5-turbo'
}

# 创建生成器
generator = PWACodeGenerator(config)

# 模板数据
template_data = {
    "module_name": "fit_analysis",
    "data_path": "/path/to/data",
    "parameters": {"max_iterations": 1000}
}

# 生成代码
code = generator.generate_fit_code(template_data)
print(code)
```

### 替代原有 Jinja2 流程

```python
from agent.llm_create_control import LLMCreateCode
import json

# 加载配置
with open("config/generator_kk.json", encoding='utf-8') as f:
    dict_json = json.loads(f.read())

# Agent 配置
agent_config = {
    'openai_api_key': 'your-api-key',
    'model': 'gpt-3.5-turbo'
}

# 创建 LLM 版本的代码生成器
llm_creator = LLMCreateCode(dict_json, agent_config)

# 生成所有代码
llm_creator.generate_all()
```

## 核心组件

### 1. BaseAgent (base.py)

提供所有 Agent 的基础抽象类，包含：
- 日志记录功能
- 代码验证接口
- 文件保存功能

### 2. OpenAIClient (openai_client.py)

OpenAI API 客户端，支持：
- 标准聊天完成请求
- Function Calling 功能
- 响应验证和内容提取
- 错误处理

### 3. CodeGenerator (code_generator.py)

核心代码生成器，特性：
- 智能代码生成
- Function Calling 格式控制
- 代码语法验证
- 批量生成支持

### 4. PWACodeGenerator

专门用于 PWA 项目的生成器，包含：
- `generate_fit_code()` - 生成拟合代码
- `generate_draw_code()` - 生成绘图代码
- `generate_tensor_code()` - 生成张量计算代码
- `batch_generate()` - 批量生成

## Function Calling 格式

系统使用严格的 Function Calling 来控制 LLM 输出格式：

```json
{
  "name": "generate_python_code",
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
      }
    },
    "required": ["code", "imports", "description"]
  }
}
```

## 测试

运行测试脚本：

```bash
cd agent
python test_agent.py
```

测试包括：
- OpenAI 客户端连接测试
- Function Calling 功能测试
- 文件操作测试
- 代码生成和验证测试

## 优势对比

| 特性 | Jinja2 模板 | LLM Agent |
|------|-------------|-----------|
| 灵活性 | 模板固定 | 智能适配 |
| 代码质量 | 依赖模板质量 | AI 优化 |
| 错误处理 | 基础检查 | 智能验证 |
| 维护成本 | 需要维护模板 | 自适应 |
| 自然语言支持 | 无 | 支持 |

## 错误处理

系统包含完善的错误处理机制：

- `CodeGenerationError` - 代码生成异常
- `ValidationError` - 代码验证异常  
- `OpenAIError` - API 调用异常

## 扩展说明

要添加新的代码类型生成：

1. 在 `code_generator.py` 中添加新的生成方法
2. 更新 Function Calling 定义
3. 在 `llm_create_control.py` 中添加对应的调用逻辑
4. 编写相应的测试用例

## 注意事项

1. 确保 OpenAI API 密钥有效且有足够的配额
2. 生成的代码建议进行人工审查
3. 大型代码生成可能需要较长时间
4. 建议使用 gpt-3.5-turbo 以上的模型以获得更好效果