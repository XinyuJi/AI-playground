import os
import re
import json
from typing import Dict

from openai import OpenAI

model_list = ["Qwen/Qwen3-4B-fast","Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "google/gemma-2-2b-it","google/gemma-2-9b-it","google/gemma-2-27b-it","Qwen/Qwen3-235B-A22B","deepseek-ai/DeepSeek-R1"]
no_think_model_list = ["Qwen/Qwen3-4B-fast","Qwen/Qwen3-14B", "Qwen/Qwen3-32B"] # Do not enable think mode for these models

# Use api key either explicitly or from environment variable
OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNzM0NTcxOTkyODExNDUwODI4MiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNTEyNzc4NSwidXVpZCI6IjdiMzNiYWRhLTlkYzYtNDZlYi1iZTNlLTA1OTRiNzJmYTJmNyIsIm5hbWUiOiJleHBfMSIsImV4cGlyZXNfYXQiOiIyMDMwLTA1LTE2VDAyOjA5OjQ1KzAwMDAifQ.sOWnRk0abLtmZtZre5e6pkM74VxLn6odUmBGbACZZ7o"
# 2nd api key
# OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNzM0NTcxOTkyODExNDUwODI4MiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNTE4NzgxOCwidXVpZCI6ImNhOTgwMjY5LWFkMTAtNDI0OC1hZmI4LTdjMGZkMjU4NWQ5MyIsIm5hbWUiOiJleHBfMiIsImV4cGlyZXNfYXQiOiIyMDMwLTA1LTE2VDE4OjUwOjE4KzAwMDAifQ.nkMH6qm76IMimDYcmRGo6IDy8sLhcVrehsHCu1kK6PQ"
# print(os.getenv("OPENAI_API_KEY")) 

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=OPENAI_API_KEY,
)

def parse_completion_output(json_str: str) -> Dict[str, str]:
    """
    Parse the model output JSON string and extract the main text content.
    
    Args:
        json_str (str): The JSON string returned from `completion.to_json()`.
        include_think (bool): Whether to extract and return the <think> block separately.
    
    Returns:
        Dict[str, str]: A dictionary with keys:
            - "full_content": The full content from model
            - "think_content": The extracted <think>...</think> content (if any, only if include_think=True)
            - "main_content": The content with <think> section removed (if include_think=True), or same as full_content
    """
    try:
        data = json.loads(json_str)
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {
            "error": f"Failed to parse JSON: {e}"
        }

    result = {
        "full_content": content
    }


    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    main_content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

    result["think_content"] = think_content
    result["main_content"] = main_content


    return result

for each_model in model_list:
    print(f"Model: {each_model}")
    
    message_string = "你好，你是谁？"
    message_to_send = message_string + '/no_think' if each_model in no_think_model_list else message_string

    completion = client.chat.completions.create(
        model=each_model,
        messages=[
            {
                "role": "user",
                "content": message_to_send
            }
        ],
        temperature=0.9
    )
    
    output = parse_completion_output(completion.to_json())
    print(json.dumps(output, indent=2, ensure_ascii=False))  # 支持中文显示


