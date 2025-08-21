# 导入标准库中的 tomllib 模块
import tomllib

try:
    # 使用 'rb' (二进制读取) 模式打开文件
    with open("resonances_config.toml", "rb") as f:
        # 使用 tomllib.load() 解析文件
        config_data = tomllib.load(f)

    # 打印解析后的数据（会是一个字典）
    print("--- 配置文件内容 ---")
    # print(f"应用标题: {config_data['metadata']}")
    print(f"共振态列表: {config_data['resonances']['f980']}")

except FileNotFoundError:
    print("错误：配置文件 config.toml 未找到。")
except tomllib.TOMLDecodeError as e:
    print(f"错误：解析 TOML 文件失败: {e}")