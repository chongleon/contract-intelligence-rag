import json
import os
import re
from dashscope import Generation
import dashscope
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ===== explain 输出 schema =====
EXPLAIN_SCHEMA = {
    "waiting_period_explanation": "",
    "coverage_explanation": "",
    "suitable_for": "",
    "risk_warning": ""
}

SYSTEM_PROMPT = """
你是保险条款通俗化解读助手。

任务：
根据给定的结构化保险信息，用普通消费者能理解的语言解释。

要求：
1. 不使用专业术语
2. 每段不超过100字
3. 输出必须是合法JSON
4. 不要输出多余文字
"""

def build_explain_prompt(structured_json: dict) -> str:
    return f"""
下面是某保险产品的结构化信息：

{json.dumps(structured_json, ensure_ascii=False)}

请生成通俗化解释。

输出格式：
{json.dumps(EXPLAIN_SCHEMA, ensure_ascii=False)}
"""

def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```.*?\n", "", text)
        text = text.replace("```", "")
    return text


def llm_explain(structured_json: dict) -> dict:
    prompt = build_explain_prompt(structured_json)

    response = Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = response["output"]["text"]
    content = clean_json(content)

    try:
        result = json.loads(content)
    except:
        result = EXPLAIN_SCHEMA

    return result

if __name__ == "__main__":
    sample_structured = {
        "product_name": "泰康医疗险",
        "insurance_type": "医疗险",
        "coverage_period": "一年",
        "coverage_amount": "100万元",
        "waiting_period": "90天",
        "exclusions": ["酒后驾驶", "既往症"]
    }

    explain_result = llm_explain(sample_structured)
    print(json.dumps(explain_result, ensure_ascii=False, indent=2))