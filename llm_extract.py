import json
import re
import os
from dashscope import Generation
import dashscope
from dotenv import load_dotenv

# ========= 加载环境变量 =========
load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope.api_key:
    raise ValueError("请先在 .env 文件中设置 DASHSCOPE_API_KEY")

# ========= 固定 Schema =========
SCHEMA_TEMPLATE = {
    "product_name": "",
    "insurance_type": "",
    "coverage_period": "",
    "coverage_amount": "",
    "waiting_period": "",
    "exclusions": []
}

SYSTEM_PROMPT = """
你是保险条款结构化抽取助手。
必须输出合法JSON。
不允许输出解释。
字段缺失用空字符串。
exclusions 必须是数组。
不要输出markdown。
"""

def build_prompt(text: str) -> str:
    return f"""
请从以下保险条款中抽取信息，并严格按JSON格式输出：

字段结构：
{json.dumps(SCHEMA_TEMPLATE, ensure_ascii=False)}

条款内容：
{text}
"""

def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```.*?\n", "", text)
        text = text.replace("```", "")
    return text

def llm_extract(text: str) -> dict:
    prompt = build_prompt(text)

    response = Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    print("完整response：", response)   

    if response is None:
        print("API调用失败，response为None")
        return SCHEMA_TEMPLATE

    if "output" not in response:
        print("返回结构异常：", response)
        return SCHEMA_TEMPLATE

    content = response["output"]["text"]
    content = clean_json(content)

    try:
        result = json.loads(content)
    except Exception as e:
        print("JSON解析失败:", e)
        result = SCHEMA_TEMPLATE

    return result


# ====== 单独测试用 ======
if __name__ == "__main__":
    sample_text = "本保险产品为泰康医疗险，保险期间一年，保额100万元，等待期90天。"
    result = llm_extract(sample_text)
    print(result)