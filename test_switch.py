from .llm_factory import LLMFactory
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
import yaml
from pathlib import Path

def update_config(provider: str):
    """更新配置文件中的 provider"""
    config_path = Path(__file__).parent.parent.parent / "conf.yaml"
    
    # 读取现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新 provider
    config['active_provider'] = provider
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

def test_provider(provider: str):
    """测试指定的 provider"""
    print(f"\n=== 测试 {provider} ===")
    
    # 更新配置
    update_config(provider)
    
    # 创建 LLM 实例
    llm = LLMFactory.create()
    
    # 测试基本调用
    prompt = "Say hello and introduce yourself briefly."
    print("\n1. 测试基本调用:")
    print(f"Prompt: {prompt}")
    response = llm.call(prompt)
    print(f"Response: {response}")
    
    # 测试工具调用
    print("\n2. 测试工具调用:")
    def search_web(query: str) -> str:
        return f"Searching for: {query}"
        
    tools = [
        Tool(
            name="web_search",
            description="Search the web for information",
            func=search_web
        )
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        HumanMessage(content="Search for information about Python programming")
    ]
    
    response = llm_with_tools.invoke(messages)
    print(f"Response: {response.content}")

def main():
    """主测试函数"""
    # 测试 Gemini
    test_provider("gemini")
    
    # 测试 OpenAI
    test_provider("openai")

if __name__ == "__main__":
    main() 