import os #interact with the operating system
from dotenv import load_dotenv #load .env file
import google.generativeai as genai #google's api for gemini port
from google.generativeai import GenerativeModel #google's api for generative ai
from langchain_core.messages import BaseMessage #langchain's message class
from langchain_core.tools import BaseTool #langchain's tool class
from langchain_core.messages import HumanMessage, AIMessage #langchain's message class
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


# 加载环境变量
load_dotenv()


class GeminiLLM: 
    def __init__(self, config_path: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None):
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 确定使用的模型和API密钥
        self.model_name = model_name or self.config.get("gemini", {}).get("model", "gemini-2.5-pro-preview-05-20")
        self.api_key = api_key or self.config.get("gemini", {}).get("api_key") or os.environ.get("GOOGLE_API_KEY")
        
        print(f"\nDebug - Gemini LLM:")
        print(f"Model name: {self.model_name}")
        print(f"API key length: {len(self.api_key) if self.api_key else 0}")
        print(f"API key prefix: {self.api_key[:4]}..." if self.api_key else "No API key")
        print(f"Raw API key from config: {self.config.get('gemini', {}).get('api_key')}")
        print(f"Raw API key from env: {os.environ.get('GOOGLE_API_KEY')}")
        print(f"Final API key used: {self.api_key}")
        
        if not self.api_key:
            raise ValueError("Gemini API key not found in config or environment variables.")
            
        if self.config.get("active_provider") != "gemini":
            raise ValueError("Config must be set to use Gemini provider")
            
        # 配置 API
        genai.configure(api_key=self.api_key)
        
        try:
            print("\nAvailable models:")
            for m in genai.list_models():
                print(f"- {m.name}")
        except Exception as e:
            print(f"Warning: Could not list available models: {str(e)}")
            
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"\nSuccessfully initialized model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
            
        self.tools = []
        
        # 加载其他配置
        self.temperature = self.config.get("gemini", {}).get("temperature", 0.7)
        self.max_tokens = self.config.get("gemini", {}).get("max_tokens", 1000)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 尝试在项目根目录找到配置文件
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "conf.yaml"
            
        if not os.path.exists(config_path):
            # 如果找不到配置文件，返回默认配置
            return {
                "active_provider": "gemini",
                "gemini": {
                    "model": "gemini-2.5-pro-preview-05-20",
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 处理环境变量替换
        if "${GOOGLE_API_KEY}" in str(config.get("gemini", {}).get("api_key", "")):
            config["gemini"]["api_key"] = os.getenv("GOOGLE_API_KEY")
            
        if "${OPENAI_API_KEY}" in str(config.get("openai", {}).get("api_key", "")):
            config["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
            
        return config

    def call(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")

    def bind_tools(self, tools: list[BaseTool]):
        self.tools = tools
        return self
    
    def invoke(self, messages: list[BaseMessage], **kwargs):
        # 1. 拼接成 Gemini 能理解的 prompt
        prompt = ""
        
        # 添加系统提示词
        if self.tools:
            system_prompt = (
                "You are an AI assistant with access to the following tools:\n\n"
                "Available tools:\n"
            )
            for tool in self.tools:
                system_prompt += f"- {tool.name}: {tool.description}\n"
            system_prompt += (
                "\nIMPORTANT INSTRUCTIONS FOR TOOL USAGE:\n"
                "1. When you need to use a tool, ONLY respond with the exact format: tool_name(\"argument\")\n"
                "2. Do not include any other text in your response when calling a tool\n"
                "3. Only use the tool names listed above\n"
                "4. Arguments must be in quotes\n\n"
                "Example: web_search(\"python programming\")\n\n"
            )
            prompt = system_prompt

        # 添加对话历史
        for m in messages:
            if isinstance(m, HumanMessage):
                prompt += f"Human: {m.content}\n"
            elif isinstance(m, AIMessage):
                prompt += f"AI: {m.content}\n"

        try:
            # 2. 调用 Gemini 模型
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
            )
            raw_text = response.text.strip()

            # 3. 判断是否需要工具调用
            tool_call = self._parse_tool_calls(raw_text)

            if tool_call:
                tool_result = self._execute_tool_call(tool_call)
                final_response = f"{raw_text}\n\n[Tool Output]: {tool_result}"
            else:
                final_response = raw_text

            # 4. 返回 LangChain 标准消息对象
            return AIMessage(content=final_response)

        except Exception as e:
            return AIMessage(content=f"[Error] {str(e)}")

    def _parse_tool_calls(self, response: str):
        """解析工具调用
        
        Args:
            response: 模型的原始响应文本
            
        Returns:
            如果是有效的工具调用，返回工具调用信息字典；否则返回 None
        """
        # 1. 检查响应是否只包含工具调用（去除空白字符）
        response = response.strip()
        if not response:
            return None
            
        # 2. 使用更严格的正则表达式匹配
        pattern = r"^(\w+)\(\"([^\"]+)\"\)$"  # 必须严格匹配整个字符串
        match = re.match(pattern, response)
        
        if not match:
            return None
            
        tool_name = match.group(1)
        arg = match.group(2)
        
        # 3. 验证工具名称是否在可用工具列表中
        if not any(tool.name == tool_name for tool in self.tools):
            return None
            
        return {
            "tool_name": tool_name,
            "args": {"query": arg}
        }

    def _execute_tool_call(self, tool_call: dict):
        """
        从绑定的 tools 中找到对应工具并执行。
        """
        tool_name = tool_call["tool_name"]
        args = tool_call["args"]

        # 在 self.tools 中寻找匹配的工具
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    return tool.run(args)
                except Exception as e:
                    return f"[Tool Error: {tool_name}] {str(e)}"

        return f"[Tool Not Found] No tool named '{tool_name}' is registered."
