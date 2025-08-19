#!/usr/bin/env python3
# coding: utf-8
"""
极易云 API 测试运行脚本
从项目根目录运行，避免导入问题
"""

if __name__ == "__main__":
    # 直接导入和运行
    import os
    os.environ['EASYTRANS_API_KEY'] = 'sk-v1-LJfF-3OKN3ZyESALnL08vWbfSOQ-MYIop/n/' 
    from agent.easytrans_example import main
    main()