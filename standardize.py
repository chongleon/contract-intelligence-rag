import re

def normalize_amount(text: str):
    if not text:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    value = float(nums[0])
    if "万" in text:
        return value * 10000
    if "亿" in text:
        return value * 100000000
    return value

def normalize_waiting(text: str):
    if not text:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    return int(nums[0])

def standardize(data: dict) -> dict:
    return {
        "product_name": data.get("product_name", ""),
        "insurance_type": data.get("insurance_type", ""),
        "coverage_period": data.get("coverage_period", ""),
        "coverage_amount_raw": data.get("coverage_amount", ""),
        "coverage_amount_value": normalize_amount(data.get("coverage_amount", "")),
        "waiting_period_raw": data.get("waiting_period", ""),
        "waiting_period_value": normalize_waiting(data.get("waiting_period", "")),
        "exclusions": data.get("exclusions", [])
    }


# ====== 单独测试 ======
if __name__ == "__main__":
    sample = {
        "coverage_amount": "100万元",
        "waiting_period": "90天"
    }
    print(standardize(sample))

def compare_policy(a: dict, b: dict) -> dict:
    result = {}

    # 保额比较
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

    return result