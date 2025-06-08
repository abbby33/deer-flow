import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

class OpenAILLM:
    def __init__(self, config_path: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None):
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 确定使用的模型和API密钥
        self.model_name = model_name or self.config.get("openai", {}).get("model", "gpt-3.5-turbo")
        self.api_key = api_key or self.config.get("openai", {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.organization = self.config.get("openai", {}).get("organization")
        
        print(f"\nDebug - OpenAI LLM:")
        print(f"Model name: {self.model_name}")
        print(f"API key length: {len(self.api_key) if self.api_key else 0}")
        print(f"API key prefix: {self.api_key[:4]}..." if self.api_key else "No API key")
        print(f"Organization: {self.organization}")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables.")
            
        if self.config.get("active_provider") != "openai":
            raise ValueError("Config must be set to use OpenAI provider")
            
        # 配置 API
        client_kwargs = {
            "api_key": self.api_key,
        }
        if self.organization:
            client_kwargs["organization"] = self.organization
            
        self.client = OpenAI(**client_kwargs)
        self.tools = []
        
        # 加载其他配置
        self.temperature = self.config.get("openai", {}).get("temperature", 0.7)
        self.max_tokens = self.config.get("openai", {}).get("max_tokens", 1000)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 尝试在项目根目录找到配置文件
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "conf.yaml"
            
        if not os.path.exists(config_path):
            # 如果找不到配置文件，返回默认配置
            return {
                "active_provider": "openai",
                "openai": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 处理环境变量替换
        if "${OPENAI_API_KEY}" in str(config.get("openai", {}).get("api_key", "")):
            config["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
            
        return config

    def call(self, prompt: str, **kwargs) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")

    def bind_tools(self, tools: list[BaseTool]):
        self.tools = tools
        return self
    
    def invoke(self, messages: list[BaseMessage], **kwargs):
        # 1. 转换消息格式
        openai_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                openai_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                openai_messages.append({"role": "assistant", "content": m.content})

        # 添加系统提示词
        if self.tools:
            system_prompt = "You are an AI assistant with access to the following tools:\n"
            for tool in self.tools:
                system_prompt += f"- {tool.name}: {tool.description}\n"
            system_prompt += "\nWhen you need to use a tool, respond with: toolname(\"argument\")\n\n"
            openai_messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            # 2. 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            raw_text = response.choices[0].message.content

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
        """找出ai想要用的tool"""
        import re
        pattern = r"(\w+)\(\"([^\"]+)\"\)"  # 匹配格式：toolname("argument")
        match = re.search(pattern, response)

        if match:
            tool_name = match.group(1)
            arg = match.group(2)
            return {
                "tool_name": tool_name,
                "args": {"query": arg}  
            }

        return None

    def _execute_tool_call(self, tool_call: dict):
        """从绑定的 tools 中找到对应工具并执行"""
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