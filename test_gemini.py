import os
from dotenv import load_dotenv
import google.generativeai as genai

# 加载环境变量
load_dotenv()

# 获取 API key
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API key: {api_key}")

# 配置 API
genai.configure(api_key=api_key)

# 获取可用模型
# print("\n可用的模型:")
# for m in genai.list_models():
#     print(m.name)

# 创建模型实例
model = genai.GenerativeModel('gemini-pro')

# 测试生成
print("\n测试生成:")
response = model.generate_content("Say hello!")
print(response.text) 