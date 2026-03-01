"""RAG 检索主流程实现。"""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

import chromadb
import dashscope
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from dashscope import TextEmbedding
from dotenv import load_dotenv
from pypdf import PdfReader

logger = logging.getLogger(__name__)
if not logger.handlers:
	handler = logging.StreamHandler()
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
logger.setLevel(logging.INFO)


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf"
VECTOR_DIR = BASE_DIR / "vector_store"
MANIFEST_PATH = VECTOR_DIR / "manifest.json"
COLLECTION_NAME = "insurance_clauses"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_EMBEDDING_MODEL = os.getenv("RAG_DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")

dashscope.api_key = DASHSCOPE_API_KEY
if not dashscope.api_key:
	raise RuntimeError("请在环境变量或 .env 中配置 DASHSCOPE_API_KEY")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 5

_collection: Collection | None = None


def rag_pipeline(query: str) -> list:
	"""面向外部的检索函数，输入 query 输出最相关的段落列表。"""

	query = (query or "").strip()
	if not query:
		raise ValueError("query 不能为空")

	collection = _ensure_collection_ready()
	logger.info("执行向量检索: %s", query)
	results = collection.query(query_texts=[query], n_results=DEFAULT_TOP_K)

	documents = results.get("documents", [[]])[0]
	metadatas = results.get("metadatas", [[]])[0]
	distances = results.get("distances", [[]])[0]

	response = []
	for doc, meta, distance in zip(documents, metadatas, distances):
		if not doc:
			continue
		similarity = 1 - distance if distance is not None else None
		response.append({
			"content": doc,
			"score": round(similarity, 4) if similarity is not None else None,
			"source": meta.get("source") if meta else None,
			"chunk_id": meta.get("chunk_id") if meta else None,
		})

	return response


def _ensure_collection_ready() -> Collection:
	"""确保向量库存续可用，如需则重新构建。"""

	global _collection

	VECTOR_DIR.mkdir(parents=True, exist_ok=True)
	client = chromadb.PersistentClient(path=str(VECTOR_DIR))
	embedding_fn = _build_embedding_function()

	state_signature = _collect_state_signature()
	manifest = _load_manifest()
	needs_rebuild = state_signature != manifest or not _collection_exists(client)

	if needs_rebuild:
		logger.info("检测到条款文件更新，开始重新构建向量库")
		_rebuild_collection(client, embedding_fn)
		_save_manifest(state_signature)
	elif _collection is None:
		_collection = client.get_collection(
			name=COLLECTION_NAME,
			embedding_function=embedding_fn,
		)

	if _collection is None:
		_collection = client.get_collection(
			name=COLLECTION_NAME,
			embedding_function=embedding_fn,
		)

	return _collection


def _build_embedding_function():
	"""按照配置选择 embedding provider。"""

	return _build_dashscope_embedding_function()


def _build_dashscope_embedding_function():
	"""创建 DashScope Embedding 函数。"""

	return DashscopeEmbeddingFunction(
		model=DASHSCOPE_EMBEDDING_MODEL,
		text_embedding_client=TextEmbedding,
	)


def _rebuild_collection(client: chromadb.PersistentClient, embedding_fn) -> None:
	"""重新创建 collection 并批量写入 chunk 数据。"""

	global _collection

	try:
		client.delete_collection(COLLECTION_NAME)
	except NotFoundError:
		pass

	collection = client.create_collection(
		name=COLLECTION_NAME,
		embedding_function=embedding_fn,
		metadata={"description": "保险条款段落向量库"},
	)

	documents, metadatas, ids = _prepare_documents()
	if not documents:
		raise RuntimeError("未在 pdf 目录中找到可用文本，无法构建向量库")

	collection.add(
		ids=ids,
		documents=documents,
		metadatas=metadatas,
	)

	_collection = collection


def _prepare_documents() -> tuple[List[str], List[Dict[str, str]], List[str]]:
	"""读取 PDF、切块并输出 Chroma 需要的字段。"""

	documents: List[str] = []
	metadatas: List[Dict[str, str]] = []
	ids: List[str] = []

	if not PDF_DIR.exists():
		raise FileNotFoundError(f"未找到 PDF 目录: {PDF_DIR}")

	for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
		text = _extract_text(pdf_path)
		chunks = _chunk_text(text)
		for idx, chunk in enumerate(chunks):
			chunk_id = f"{pdf_path.stem}_{idx}"
			documents.append(chunk)
			metadatas.append({
				"source": pdf_path.name,
				"chunk_id": chunk_id,
			})
			ids.append(chunk_id)

	return documents, metadatas, ids


def _extract_text(pdf_path: Path) -> str:
	"""将 PDF 文本抽出为单一字符串。"""

	reader = PdfReader(str(pdf_path))
	pages = []
	for page in reader.pages:
		content = page.extract_text() or ""
		content = content.strip()
		if content:
			pages.append(content)
	full_text = "\n".join(pages)
	logger.info("完成 PDF 解析: %s", pdf_path.name)
	return full_text


def _chunk_text(text: str) -> List[str]:
	"""按固定字符长度切块，重叠保证语义连续。"""

	if not text:
		return []

	chunks: List[str] = []
	start = 0
	text_length = len(text)
	while start < text_length:
		end = min(start + CHUNK_SIZE, text_length)
		chunks.append(text[start:end])
		if end == text_length:
			break
		next_start = end - CHUNK_OVERLAP
		if next_start <= start:
			next_start = end
		start = next_start
	return chunks


def _collect_state_signature() -> Dict[str, object]:
	"""综合记录 pdf 变更与 embedding 配置，用于判断是否需重建。"""

	return {
		"files": _collect_file_signature(),
		"embedding": {
			"provider": "dashscope",
			"model_name": DASHSCOPE_EMBEDDING_MODEL,
		},
	}


def _collect_file_signature() -> Dict[str, Dict[str, float]]:
	"""生成当前 pdf 目录下文件的指纹信息。"""

	signature: Dict[str, Dict[str, float]] = {}
	if not PDF_DIR.exists():
		return signature

	for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
		stat = pdf_path.stat()
		signature[str(pdf_path.name)] = {
			"mtime": stat.st_mtime,
			"size": stat.st_size,
		}
	return signature


def _load_manifest() -> Dict[str, object]:
	"""读取上一次构建的指纹信息，并兼容旧版本结构。"""

	if not MANIFEST_PATH.exists():
		return {}
	with MANIFEST_PATH.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		return {}
	if "files" not in data:
		return {"files": data}
	if "embedding" not in data:
		data["embedding"] = {}
	return data


def _save_manifest(signature: Dict[str, object]) -> None:
	"""持久化当前指纹及 embedding 配置。"""

	with MANIFEST_PATH.open("w", encoding="utf-8") as f:
		json.dump(signature, f, ensure_ascii=False, indent=2)


def _collection_exists(client: chromadb.PersistentClient) -> bool:
	"""判断指定 collection 是否存在。"""

	try:
		client.get_collection(name=COLLECTION_NAME)
		return True
	except NotFoundError:
		return False


class DashscopeEmbeddingFunction:
	"""将 DashScope TextEmbedding 封装为 Chroma 兼容的 embedding function。"""

	def __init__(self, model: str, text_embedding_client, batch_size: int = 8):
		self.model = model
		self.batch_size = batch_size
		self._client = text_embedding_client
		self._name = f"dashscope-{self.model}"

	def name(self) -> str:
		"""Chroma 期望的 embedding 函数名称，用于冲突检测。"""

		return self._name

	def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: D401 - Chroma 接口约定
		if not input:
			return []

		clean_texts = [text if isinstance(text, str) else str(text) for text in input]
		vectors: List[List[float]] = []
		for start in range(0, len(clean_texts), self.batch_size):
			batch = clean_texts[start:start + self.batch_size]
			response = self._client.call(model=self.model, input=batch)
			if response.status_code != HTTPStatus.OK:
				raise RuntimeError(
					f"DashScope embedding 接口调用失败: {response.code} {response.message}"
				)
			output = getattr(response, "output", None)
			if not output:
				raise RuntimeError("DashScope embedding 响应缺少 output 字段")
			batch_embeddings = output.get("embeddings", [])
			batch_embeddings = sorted(batch_embeddings, key=lambda item: item.get("text_index", 0))
			if len(batch_embeddings) != len(batch):
				raise RuntimeError("DashScope embedding 返回数量与输入不一致")
			for item in batch_embeddings:
				vector = item.get("embedding")
				if vector is None:
					raise RuntimeError("DashScope embedding 返回缺少 embedding 字段")
				vectors.append(list(vector))

		return vectors

	def embed_documents(self, input: List[str]) -> List[List[float]]:
		"""Chroma 兼容接口，调用与 __call__ 相同的逻辑。"""

		return self(input)

	def embed_query(self, input: str) -> List[List[float]]:
		"""针对单条 query 的 embedding，返回二维列表。"""

		return self([input])

