from llm_extract import llm_extract
from llm_explain import llm_explain
from standardize import standardize, compare_policy
import json

# ===== 两个测试文本 =====
text_a = "本保险为泰康医疗险，保险期间一年，保额100万元，等待期90天。"
text_b = "本保险为平安医疗险，保险期间一年，保额80万元，等待期60天。"

# ===== 第一步：结构化抽取 =====
json_a = llm_extract(text_a)
json_b = llm_extract(text_b)

# ===== 第二步：标准化 =====
std_a = standardize(json_a)
std_b = standardize(json_b)

# ===== 第三步：通俗化解释 =====
explain_a = llm_explain(json_a)
explain_b = llm_explain(json_b)

# ===== 第四步：对比 =====
compare_result = compare_policy(std_a, std_b)

# ===== 打印结果 =====
print("\n===== A结构化结果 =====")
print(json.dumps(json_a, ensure_ascii=False, indent=2))

print("\n===== B结构化结果 =====")
print(json.dumps(json_b, ensure_ascii=False, indent=2))

print("\n===== A标准化结果 =====")
print(std_a)

print("\n===== B标准化结果 =====")
print(std_b)

print("\n===== A通俗化解释 =====")
print(json.dumps(explain_a, ensure_ascii=False, indent=2))

print("\n===== B通俗化解释 =====")
print(json.dumps(explain_b, ensure_ascii=False, indent=2))

print("\n===== 对比结果 =====")
print(compare_result)