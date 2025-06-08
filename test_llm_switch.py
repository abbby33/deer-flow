import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv()

# 调试信息
print("=== 详细的环境变量检查 ===")
print(f"当前目录: {os.getcwd()}")
print(f".env 文件是否存在: {os.path.exists('.env')}")
print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')}")
print(f"GOOGLE_API_KEY 长度: {len(os.getenv('GOOGLE_API_KEY') or '')}")
print("========================\n")

from src.llms.test_switch import main

if __name__ == "__main__":
    main() 