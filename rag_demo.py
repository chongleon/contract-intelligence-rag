import json

from rag_pipeline import rag_pipeline

if __name__ == "__main__":
    query = "请求赔偿时，应提交哪些索赔材料？"
    results = rag_pipeline(query)
    print(json.dumps(results, ensure_ascii=False, indent=2))