from llm_extract import llm_extract
from standardize import standardize, compare_policy

text_a = "本保险为泰康医疗险，保险期间一年，保额100万元，等待期90天。"
text_b = "本保险为平安医疗险，保险期间一年，保额80万元，等待期60天。"

json_a = llm_extract(text_a)
json_b = llm_extract(text_b)

std_a = standardize(json_a)
std_b = standardize(json_b)

compare_result = compare_policy(std_a, std_b)

print("A标准化结果:", std_a)
print("B标准化结果:", std_b)
print("对比结果:", compare_result)