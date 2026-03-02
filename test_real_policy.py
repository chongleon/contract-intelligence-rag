from pathlib import Path
from llm_extract import llm_extract
from standardize import standardize
import json

# 读取真实条款
project_root = Path(__file__).resolve().parent
txt_path = project_root / "data" / "raw" / "covid.txt"

with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()

print("文本长度：", len(text))

# 1️⃣ 结构化
json_result = llm_extract(text)
print("\n===== 原始抽取结果 =====")
print(json.dumps(json_result, ensure_ascii=False, indent=2))

# 2️⃣ 标准化
std_result = standardize(json_result)
print("\n===== 标准化结果 =====")
print(json.dumps(std_result, ensure_ascii=False, indent=2))