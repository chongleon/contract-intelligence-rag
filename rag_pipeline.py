"""RAG 检索主流程实现。"""

from __future__ import annotations

import json
import logging
import os
import re
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

import chromadb
import dashscope
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from dashscope import Generation, TextEmbedding
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
if not logger.handlers:
	handler = logging.StreamHandler()
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
logger.setLevel(logging.INFO)


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "data" / "raw"
VECTOR_DIR = BASE_DIR / "vector_store"
MANIFEST_PATH = VECTOR_DIR / "manifest.json"
COLLECTION_NAME = "insurance_clauses"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_EMBEDDING_MODEL = os.getenv("RAG_DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
DASHSCOPE_COMPLETION_MODEL = os.getenv("RAG_DASHSCOPE_COMPLETION_MODEL", "qwen-turbo")

dashscope.api_key = DASHSCOPE_API_KEY
if not dashscope.api_key:
	raise RuntimeError("请在环境变量或 .env 中配置 DASHSCOPE_API_KEY")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 5
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1200"))
_HEADING_PATTERNS = [
	re.compile(r"^第[一二三四五六七八九十百千万零]+[章节条款]?"),
	re.compile(r"^第?\d+[章节条款]?"),
	re.compile(r"^(\d+\.|\d+．|\d+、)"),
	re.compile(r"^[（(]?[一二三四五六七八九十]+[）).、]"),
	re.compile(r"^[A-Za-z]{1,3}[\.、]\s*"),
]
RAG_SYSTEM_PROMPT = (
	"你是保险条款问答助手。严格依赖提供的条款片段回答问题，"
	"如信息不足必须明确说明无法回答，不要编造。输出简洁中文并保留必要条理。"
)


_collection: Collection | None = None


def rag_pipeline(query: str) -> Dict[str, object]:
	"""执行 RAG：检索相关段落并使用 LLM 生成回答。"""

	query = (query or "").strip()
	if not query:
		raise ValueError("query 不能为空")

	collection = _ensure_collection_ready()
	logger.info("执行向量检索: %s", query)
	results = collection.query(query_texts=[query], n_results=DEFAULT_TOP_K)

	documents = results.get("documents", [[]])[0]
	metadatas = results.get("metadatas", [[]])[0]
	distances = results.get("distances", [[]])[0]

	contexts: List[Dict[str, object]] = []
	for doc, meta, distance in zip(documents, metadatas, distances):
		if not doc:
			continue
		similarity = 1 - distance if distance is not None else None
		context = {
			"content": doc,
			"score": round(similarity, 4) if similarity is not None else None,
			"source": meta.get("source") if meta else None,
			"chunk_id": meta.get("chunk_id") if meta else None,
		}
		contexts.append(context)

	if not contexts:
		return {
			"answer": "未检索到相关条款，无法回答该问题。",
			"contexts": [],
		}

	answer = _generate_rag_answer(query, contexts)
	return {
		"answer": answer,
		"contexts": contexts,
	}


def _generate_rag_answer(query: str, contexts: List[Dict[str, object]]) -> str:
	"""调用 DashScope LLM，基于检索上下文生成回答。"""

	context_blocks: List[str] = []
	for idx, item in enumerate(contexts, start=1):
		source = item.get("source") or "unknown"
		chunk_id = item.get("chunk_id") or f"chunk_{idx}"
		score = item.get("score")
		prefix = f"[片段{idx}] 来源: {source} | chunk_id: {chunk_id}"
		if score is not None:
			prefix += f" | 相似度: {score}"
		context_blocks.append(f"{prefix}\n{item.get('content', '')}")

	context_text = "\n\n".join(context_blocks)
	user_prompt = (
		"请仅依据以下检索到的保险条款片段回答用户问题，必要时在句末使用"
		"【chunk_id】标注来源；若无法回答请明确说明。\n"
		f"用户问题：{query}\n"
		"检索片段：\n"
		f"{context_text}"
	)

	try:
		response = Generation.call(
			model=DASHSCOPE_COMPLETION_MODEL,
			messages=[
				{"role": "system", "content": RAG_SYSTEM_PROMPT},
				{"role": "user", "content": user_prompt},
			],
			temperature=0.2,
			max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
		)
	except Exception as exc:
		raise RuntimeError("DashScope 生成回答失败") from exc

	output = response.get("output") if isinstance(response, dict) else None
	if not output:
		raise RuntimeError("DashScope 响应缺少 output 字段")
	content = output.get("text")
	if not content:
		raise RuntimeError("DashScope 响应缺少 text 字段")

	return content.strip()


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
		raise RuntimeError("未在 data/raw 目录中找到可用文本，无法构建向量库")

	collection.add(
		ids=ids,
		documents=documents,
		metadatas=metadatas,
	)

	_collection = collection


def _prepare_documents() -> tuple[List[str], List[Dict[str, str]], List[str]]:
	"""读取文本文件、切块并输出 Chroma 需要的字段。"""

	documents: List[str] = []
	metadatas: List[Dict[str, str]] = []
	ids: List[str] = []

	if not TEXT_DIR.exists():
		raise FileNotFoundError(f"未找到文本目录: {TEXT_DIR}")

	for text_path in sorted(TEXT_DIR.glob("*.txt")):
		text = _read_text_file(text_path)
		chunks = _chunk_text(text)
		for idx, chunk in enumerate(chunks):
			chunk_id = f"{text_path.stem}_{idx}"
			documents.append(chunk)
			metadatas.append({
				"source": text_path.name,
				"chunk_id": chunk_id,
			})
			ids.append(chunk_id)

	return documents, metadatas, ids


def _read_text_file(file_path: Path) -> str:
	"""读取 txt 文件内容并规范化换行。"""

	content = file_path.read_text(encoding="utf-8")
	content = content.replace("\r\n", "\n").replace("\r", "\n")
	logger.info("读取文本: %s", file_path.name)
	return content


def _chunk_text(text: str) -> List[str]:
	"""结合段落/标题的语义切块，尽量保持业务条款完整性。"""

	paragraphs = _split_paragraphs(text)
	if not paragraphs:
		return []

	chunks: List[str] = []
	current: List[str] = []
	current_len = 0

	def flush_current() -> None:
		nonlocal current, current_len
		chunk = "\n\n".join(seg.strip() for seg in current if seg.strip())
		if chunk:
			chunks.append(chunk)
		current = []
		current_len = 0

	for para in paragraphs:
		if _looks_like_heading(para) and current:
			flush_current()
		if _looks_like_heading(para):
			current = [para]
			current_len = len(para)
			continue

		segments = _shard_long_text(para)
		for segment in segments:
			if not segment:
				continue
			projected = current_len + (2 if current else 0) + len(segment)
			if current and projected > CHUNK_SIZE:
				flush_current()
			current.append(segment)
			current_len = len("\n\n".join(current))

	if current:
		flush_current()

	return chunks


def _split_paragraphs(text: str) -> List[str]:
	"""按照空行拆分段落，并清理多余空白。"""

	if not text:
		return []
	normalized = text.replace("\r\n", "\n").replace("\r", "\n")
	parts = re.split(r"\n\s*\n+", normalized)
	return [part.strip() for part in parts if part.strip()]


def _looks_like_heading(paragraph: str) -> bool:
	"""判断段落是否类似条款标题，用于切块边界。"""

	if not paragraph:
		return False
	text = paragraph.strip()
	if len(text) > 120:
		return False
	if text.endswith(("。", "；", "，")):
		return False
	return any(pattern.match(text) for pattern in _HEADING_PATTERNS)


def _shard_long_text(paragraph: str) -> List[str]:
	"""将超长段落按句子/字数进一步切分。"""

	paragraph = paragraph.strip()
	if not paragraph:
		return []
	if len(paragraph) <= CHUNK_SIZE:
		return [paragraph]

	sentences = _split_sentences(paragraph)
	segments: List[str] = []
	buffer = ""
	for sentence in sentences:
		sentence = sentence.strip()
		if not sentence:
			continue
		if len(sentence) > CHUNK_SIZE:
			for start in range(0, len(sentence), CHUNK_SIZE):
				segments.append(sentence[start:start + CHUNK_SIZE])
			continue
		if len(buffer) + len(sentence) <= CHUNK_SIZE:
			buffer += sentence
		else:
			if buffer:
				segments.append(buffer)
			buffer = sentence
	if buffer:
		segments.append(buffer)
	return segments


def _split_sentences(text: str) -> List[str]:
	"""基于中文标点的粗粒度句子切分。"""

	separators = set("。！？!?；;\n")
	sentences: List[str] = []
	current: List[str] = []
	for char in text:
		current.append(char)
		if char in separators:
			sentence = "".join(current).strip()
			if sentence:
				sentences.append(sentence)
			current = []
	if current:
		sentence = "".join(current).strip()
		if sentence:
			sentences.append(sentence)
	return sentences


def _collect_state_signature() -> Dict[str, object]:
	"""综合记录文本源变更与 embedding 配置，用于判断是否需重建。"""

	return {
		"files": _collect_file_signature(),
		"embedding": {
			"provider": "dashscope",
			"model_name": DASHSCOPE_EMBEDDING_MODEL,
		},
	}


def _collect_file_signature() -> Dict[str, Dict[str, float]]:
	"""生成 data/raw 下 txt 文件的指纹信息。"""

	signature: Dict[str, Dict[str, float]] = {}
	if not TEXT_DIR.exists():
		return signature

	for text_path in sorted(TEXT_DIR.glob("*.txt")):
		stat = text_path.stat()
		signature[str(text_path.name)] = {
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

