import re
import pandas as pd


# ========= 字段标准化函数 ========

def normalize_amount(text: str) -> float:
    """
    保额标准化：提取金额数值（单位：元）
    支持格式：100万、100万元、1亿、5000元、5000
    """
    if not text:
        return None
    # 匹配数字（支持小数）
    nums = re.findall(r"\d+\.?\d*", text)
    if not nums:
        return None
    value = float(nums[0])
    # 单位转换
    if "亿" in text:
        return value * 100000000
    if "万" in text:
        return value * 10000
    return value

def normalize_waiting(text: str) -> int:
    """
    等待期标准化：提取天数
    支持格式：90天、90日、90、三十天、一百八十天
    """
    if not text:
        return None
    
    # 中文数字映射
    cn_nums = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, 
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    
    # 尝试匹配中文数字（如：三十天、一百八十天）
    cn_num_str = ""
    for char in text:
        if char in cn_nums or char in ["十", "百"]:
            cn_num_str += char
        elif cn_num_str:  # 遇到非数字字符，停止收集
            break
    
    if cn_num_str:
        # 解析中文数字
        result = 0
        i = 0
        while i < len(cn_num_str):
            char = cn_num_str[i]
            if char == "十":
                if result == 0:
                    result = 10
                else:
                    result = result * 10
            elif char == "百":
                if result == 0:
                    result = 100
                else:
                    result = result * 100
            elif char in cn_nums:
                # 检查下一个字符
                if i + 1 < len(cn_num_str):
                    next_char = cn_num_str[i + 1]
                    if next_char == "十":
                        result += cn_nums[char] * 10
                        i += 1
                    elif next_char == "百":
                        result += cn_nums[char] * 100
                        i += 1
                    else:
                        result += cn_nums[char]
                else:
                    result += cn_nums[char]
            i += 1
        
        if result > 0:
            return result
    
    # 阿拉伯数字
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[0])
    
    return None

def normalize_period(text: str) -> int:
    """
    保险期间标准化：提取年数
    支持格式：1年、一年、终身、终身保障
    """
    if not text:
        return None
    # 终身返回 -1 表示特殊值
    if "终身" in text:
        return -1
    # 中文数字映射
    cn_nums = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, 
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for cn, num in cn_nums.items():
        if cn in text:
            return num
    # 阿拉伯数字
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[0])
    return None

def standardize(data: dict) -> dict:
    """
    标准化LLM抽取的JSON数据
    输入：原始抽取结果
    输出：包含原始值和标准化值的字典
    """
    return {
        "product_name": data.get("product_name", ""),
        "insurance_type": data.get("insurance_type", ""),
        "coverage_period_raw": data.get("coverage_period", ""),
        "coverage_period_value": normalize_period(data.get("coverage_period", "")),
        "coverage_amount_raw": data.get("coverage_amount", ""),
        "coverage_amount_value": normalize_amount(data.get("coverage_amount", "")),
        "waiting_period_raw": data.get("waiting_period", ""),
        "waiting_period_value": normalize_waiting(data.get("waiting_period", "")),
        "exclusions": data.get("exclusions", []),
        "exclusions_count": len(data.get("exclusions", []))  # 新增：免责条款数量
    }


# ========= 产品对比函数 =========

def compare_policy(a: dict, b: dict) -> dict:
    """
    对比两个标准化后的保险产品，返回对比结论
    输入：两个标准化后的字典
    输出：对比结论字典
    
    对比维度：
    - 保额：越高越好（当两者都有值时）
    - 等待期：越短越好
    - 保险期间：越长越好/终身最优
    - 免责条款数：越少越好
    """
    result = {}

    # 保额比较（越高越好）- 仅当两者都有值时
    if a["coverage_amount_value"] and b["coverage_amount_value"]:
        if a["coverage_amount_value"] > b["coverage_amount_value"]:
            result["coverage_amount"] = "A更高"
        elif a["coverage_amount_value"] < b["coverage_amount_value"]:
            result["coverage_amount"] = "B更高"
        else:
            result["coverage_amount"] = "相同"

    # 等待期比较（越短越优）
    if a["waiting_period_value"] and b["waiting_period_value"]:
        if a["waiting_period_value"] < b["waiting_period_value"]:
            result["waiting_period"] = "A更优（更短）"
        elif a["waiting_period_value"] > b["waiting_period_value"]:
            result["waiting_period"] = "B更优（更短）"
        else:
            result["waiting_period"] = "相同"

    # 保险期间比较（越长越好，终身最优）
    if a["coverage_period_value"] and b["coverage_period_value"]:
        if a["coverage_period_value"] == -1 and b["coverage_period_value"] != -1:
            result["coverage_period"] = "A更优（终身）"
        elif b["coverage_period_value"] == -1 and a["coverage_period_value"] != -1:
            result["coverage_period"] = "B更优（终身）"
        elif a["coverage_period_value"] > b["coverage_period_value"]:
            result["coverage_period"] = "A更长"
        elif a["coverage_period_value"] < b["coverage_period_value"]:
            result["coverage_period"] = "B更长"
        else:
            result["coverage_period"] = "相同"

    # 免责条款数比较（越少越优）
    if a.get("exclusions_count") is not None and b.get("exclusions_count") is not None:
        if a["exclusions_count"] < b["exclusions_count"]:
            result["exclusions_count"] = "A更优（更少）"
        elif a["exclusions_count"] > b["exclusions_count"]:
            result["exclusions_count"] = "B更优（更少）"
        else:
            result["exclusions_count"] = "相同"

    return result


def generate_compare_table(a: dict, b: dict, name_a: str = "产品A", name_b: str = "产品B") -> pd.DataFrame:
    """
    生成对比表格（DataFrame格式）
    输入：两个标准化后的字典，可选产品名称
    输出：pandas DataFrame
    """
    # 格式化保额显示
    def format_amount(value):
        if value is None:
            return "-"
        if value >= 100000000:
            return f"{value/100000000:.0f}亿"
        if value >= 10000:
            return f"{value/10000:.0f}万"
        return f"{value:.0f}元"

    # 格式化保险期间显示
    def format_period(value, raw):
        if value is None:
            return raw or "-"
        if value == -1:
            return "终身"
        return f"{value}年"

    # 格式化等待期显示
    def format_waiting(value):
        if value is None:
            return "-"
        return f"{value}天"

    # 构建表格数据
    data = {
        "对比项": ["产品名称", "保险类型", "保险期间", "保额", "等待期", "免责条款数"],
        name_a: [
            a.get("product_name", "-"),
            a.get("insurance_type", "-"),
            format_period(a.get("coverage_period_value"), a.get("coverage_period_raw", "")),
            format_amount(a.get("coverage_amount_value")),
            format_waiting(a.get("waiting_period_value")),
            a.get("exclusions_count", 0)
        ],
        name_b: [
            b.get("product_name", "-"),
            b.get("insurance_type", "-"),
            format_period(b.get("coverage_period_value"), b.get("coverage_period_raw", "")),
            format_amount(b.get("coverage_amount_value")),
            format_waiting(b.get("waiting_period_value")),
            b.get("exclusions_count", 0)
        ]
    }

    # 添加对比结论行
    compare_result = compare_policy(a, b)
    
    conclusion_a = []
    conclusion_b = []
    
    if "coverage_amount" in compare_result:
        if "A更高" in compare_result["coverage_amount"]:
            conclusion_a.append("保额更优")
        elif "B更高" in compare_result["coverage_amount"]:
            conclusion_b.append("保额更优")
    
    if "waiting_period" in compare_result:
        if "A更优" in compare_result["waiting_period"]:
            conclusion_a.append("等待期更短")
        elif "B更优" in compare_result["waiting_period"]:
            conclusion_b.append("等待期更短")
    
    # 保险期间结论
    if "coverage_period" in compare_result:
        if "A更优" in compare_result["coverage_period"] or "A更长" in compare_result["coverage_period"]:
            conclusion_a.append("期间更优")
        elif "B更优" in compare_result["coverage_period"] or "B更长" in compare_result["coverage_period"]:
            conclusion_b.append("期间更优")
    
    # 免责条款数结论
    if "exclusions_count" in compare_result:
        if "A更优" in compare_result["exclusions_count"]:
            conclusion_a.append("免责更少")
        elif "B更优" in compare_result["exclusions_count"]:
            conclusion_b.append("免责更少")
    
    data["对比项"].append("对比结论")
    data[name_a].append("、".join(conclusion_a) if conclusion_a else "-")
    data[name_b].append("、".join(conclusion_b) if conclusion_b else "-")

    # 创建DataFrame
    df = pd.DataFrame(data)
    return df


# ====== 单独测试 ======
if __name__ == "__main__":
    # 测试标准化函数
    print("=== 测试标准化函数 ===")
    sample_a = {
        "product_name": "泰康医疗险",
        "insurance_type": "医疗险",
        "coverage_period": "1年",
        "coverage_amount": "100万元",
        "waiting_period": "90天",
        "exclusions": ["既往症", "整形美容"]
    }
    sample_b = {
        "product_name": "平安医疗险",
        "insurance_type": "医疗险",
        "coverage_period": "终身",
        "coverage_amount": "80万元",
        "waiting_period": "60天",
        "exclusions": ["既往症"]
    }

    std_a = standardize(sample_a)
    std_b = standardize(sample_b)

    print("A标准化结果:", std_a)
    print("B标准化结果:", std_b)

    # 测试compare_policy（返回对比结论）
    print("\n=== 测试compare_policy（对比结论）===")
    result = compare_policy(std_a, std_b)
    print("对比结论:", result)

    # 测试generate_compare_table（返回表格）
    print("\n=== 测试generate_compare_table（对比表格）===")
    table = generate_compare_table(std_a, std_b, "泰康医疗险", "平安医疗险")
    print(table)
