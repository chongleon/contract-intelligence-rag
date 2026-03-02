from pathlib import Path
from llm_extract import llm_extract
from llm_explain import llm_explain
import json
import os

# ====== 读取真实条款 ======
project_root = Path(__file__).resolve().parent
raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"

os.makedirs(processed_dir, exist_ok=True)

# 选择一个文件测试
txt_path = raw_dir / "medical_special_drug.txt"

with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()

print("文本长度：", len(text))

file_name = txt_path.stem

# ====== 结构化抽取 ======
json_result, total_tokens = llm_extract(text)

print("\n===== 原始抽取结果 =====")
print(json.dumps(json_result, ensure_ascii=False, indent=2))

# 保存抽取结果
extract_path = processed_dir / f"{file_name}_extract.json"
with open(extract_path, "w", encoding="utf-8") as f:
    json.dump(json_result, f, ensure_ascii=False, indent=2)

print(f"\n抽取结果已保存到：{extract_path}")

# ====== 自动评估 ======
expected_keys = {
    "product_name",
    "insurance_type",
    "coverage_period",
    "coverage_amount",
    "waiting_period",
    "exclusions"
}

actual_keys = set(json_result.keys())
json_stable = expected_keys == actual_keys

missing_fields = [
    k for k, v in json_result.items()
    if v == "" or v == []
]

extract_success = json_result != {}

print("\n===== 自动评估结果 =====")
print("total_tokens:", total_tokens)
print("extract_success:", extract_success)
print("json_stable:", json_stable)
print("missing_fields:", missing_fields if missing_fields else "None")

# ====== explain ======
explain_result = llm_explain(json_result)

print("\n===== explain结果 =====")
print(json.dumps(explain_result, ensure_ascii=False, indent=2))

# 保存 explain 结果
explain_path = processed_dir / f"{file_name}_explain.json"
with open(explain_path, "w", encoding="utf-8") as f:
    json.dump(explain_result, f, ensure_ascii=False, indent=2)

print(f"\nexplain结果已保存到：{explain_path}")