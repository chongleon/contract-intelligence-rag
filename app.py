
# =========================
# 固定接口（不要改函数签名）
# =========================

def structured_pipeline(text: str) -> dict:
    """
    输入：原始条款文本
    输出：固定 schema JSON
    """
    # TODO: 后续替换为真实实现
    return {
        "product_name": "示例保险",
        "insurance_type": "医疗险",
        "coverage_period": "1年",
        "coverage_amount": "100万元",
        "waiting_period": "90天",
        "exclusions": []
    }


def rag_pipeline(query: str) -> list:
    """
    输入：用户问题
    输出：RAG 检索结果列表
    """
    # TODO: 后续替换为真实实现
    return [
        {"content": "示例段落1", "score": 0.82},
        {"content": "示例段落2", "score": 0.79}
    ]