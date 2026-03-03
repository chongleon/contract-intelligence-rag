"""
批量测试脚本：验证真实条款的标准化效果
"""
from pathlib import Path
from llm_extract import llm_extract
from standardize import standardize, compare_policy, generate_compare_table
import json
import os

# ====== 配置 ======
project_root = Path(__file__).resolve().parent
raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"

os.makedirs(processed_dir, exist_ok=True)

# 获取所有txt文件
txt_files = list(raw_dir.glob("*.txt"))
print(f"找到 {len(txt_files)} 个条款文件\n")

# ====== 批量抽取和标准化 ======
all_results = []

for txt_path in txt_files:
    file_name = txt_path.stem
    print(f"{'='*50}")
    print(f"处理文件: {file_name}")
    print(f"{'='*50}")
    
    # 读取条款
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"文本长度: {len(text)} 字符")
    
    # 结构化抽取
    json_result, total_tokens = llm_extract(text)
    
    # 标准化
    std_result = standardize(json_result)
    
    # 保存结果
    all_results.append({
        "file_name": file_name,
        "raw": json_result,
        "standardized": std_result,
        "tokens": total_tokens
    })
    
    # 打印结果
    print(f"\n----- 原始抽取结果 -----")
    print(json.dumps(json_result, ensure_ascii=False, indent=2))
    
    print(f"\n----- 标准化结果 -----")
    print(f"产品名称: {std_result['product_name']}")
    print(f"保险类型: {std_result['insurance_type']}")
    print(f"保险期间: {std_result['coverage_period_raw']} -> {std_result['coverage_period_value']}")
    print(f"保额: {std_result['coverage_amount_raw']} -> {std_result['coverage_amount_value']}")
    print(f"等待期: {std_result['waiting_period_raw']} -> {std_result['waiting_period_value']}")
    print(f"免责条款数: {std_result['exclusions_count']}")
    print(f"免责条款: {std_result['exclusions']}")
    print(f"Token使用: {total_tokens}")

# ====== 汇总统计 ======
print(f"\n{'='*60}")
print("汇总统计")
print(f"{'='*60}")

# 统计各字段填充率
field_stats = {
    "product_name": 0,
    "insurance_type": 0,
    "coverage_period": 0,
    "coverage_amount": 0,
    "waiting_period": 0,
    "exclusions": 0
}

for r in all_results:
    raw = r["raw"]
    if raw.get("product_name"):
        field_stats["product_name"] += 1
    if raw.get("insurance_type"):
        field_stats["insurance_type"] += 1
    if raw.get("coverage_period"):
        field_stats["coverage_period"] += 1
    if raw.get("coverage_amount"):
        field_stats["coverage_amount"] += 1
    if raw.get("waiting_period"):
        field_stats["waiting_period"] += 1
    if raw.get("exclusions"):
        field_stats["exclusions"] += 1

print(f"\n字段填充率 (共{len(all_results)}个文件):")
for field, count in field_stats.items():
    rate = count / len(all_results) * 100
    print(f"  {field}: {count}/{len(all_results)} ({rate:.1f}%)")

# ====== 产品对比示例 ======
print(f"\n{'='*60}")
print("产品对比示例")
print(f"{'='*60}")

if len(all_results) >= 2:
    a = all_results[4]["standardized"]
    b = all_results[5]["standardized"]
    name_a = all_results[4]["file_name"]
    name_b = all_results[5]["file_name"]
    
    print(f"\n对比: {name_a} vs {name_b}")
    
    # 对比结论
    compare_result = compare_policy(a, b)
    print(f"\n对比结论: {compare_result}")
    
    # 对比表格
    table = generate_compare_table(a, b, name_a, name_b)
    print(f"\n对比表格:")
    print(table)

# ====== 保存所有结果 ======
output_path = processed_dir / "all_standardized.json"
with open(output_path, "w", encoding="utf-8") as f:
    # 转换为可序列化格式
    save_data = []
    for r in all_results:
        save_data.append({
            "file_name": r["file_name"],
            "raw": r["raw"],
            "standardized": r["standardized"],
            "tokens": r["tokens"]
        })
    json.dump(save_data, f, ensure_ascii=False, indent=2)

print(f"\n所有结果已保存到: {output_path}")
