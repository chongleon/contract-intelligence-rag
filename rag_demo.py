import json

from rag_pipeline import rag_pipeline

if __name__ == "__main__":
    query = "什么情况下不再接受续保？"
    results = rag_pipeline(query)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    