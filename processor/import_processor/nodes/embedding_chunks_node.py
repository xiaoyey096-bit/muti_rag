import hashlib
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from muti_rag.processor.import_processor.base import BaseNode, setup_logging, T
from muti_rag.processor.import_processor.state import ImportGraphState
from muti_rag.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.core.paths import get_import_temp_dir

_CACHE_FILE = "embedding_cache.json"


class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 校验state的chunks
        self.log_step("step1", "校验chunks的数据结构")
        validated_chunks = self._validate_state(state)

        # 2. 获取嵌入模型
        self.log_step("step2", "获取BGE-M3嵌入模型客户端")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败,原因:{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败,原因:{str(e)}", node_name=self.name)

        # 3. 批量嵌入
        # 3.1 获取批量阈值
        batch_size = self.config.embedding_batch_size

        # 3.2 获取chunks的总数
        total = len(validated_chunks)

        # 3.3 向量缓存：按 chunk 内容哈希复用，重试/重导不重复计费
        file_dir = state.get("file_dir", "")
        cache = self._load_cache(file_dir)

        # 3.4 遍历
        final_chunks = []
        for index in range(0, total, batch_size):
            # 获取当前这一批
            bath_chunks = validated_chunks[index:index + batch_size]
            # 获取当前这个一批的最后一个编号
            batch_end = index + len(bath_chunks)
            self.logger.info(f"嵌入批次 [{index + 1}-{batch_end}] / {total}")
            current_chunks = self._embed_chunks(bath_chunks, embed_model, cache)
            final_chunks.extend(current_chunks)
            self._save_cache(file_dir, cache)

        # 4. 更新state的chunks
        state['chunks'] = final_chunks
        # 5. 返回
        return state

    def _validate_state(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        # 1. 获取chunks
        chunks = state.get('chunks')

        # 2. 校验chunks类型
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 3. 遍历检验每一个chunk的类型
        for index, chunk in enumerate(chunks):

            # 校验单个chunk
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"[chunk_{index + 1}] 类型和期望的类型不匹配，实际的类型{type(chunk).__name__}",
                    node_name=self.name)

        # 4. 返回chunks
        return chunks

    def _embed_chunks(self, bath_chunks: List[Dict[str, Any]], embed_model: BGEM3EmbeddingFunction,
                      cache: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        批量嵌入chunks（带内容哈希缓存：命中直接复用向量）
        Args:
            bath_chunks: 批量chunks
            embed_model: 嵌入模型
            cache: chunk_hash -> {"dense": [...], "sparse": {...}}

        Returns:

        """
        cache = cache if cache is not None else {}

        # 1. 获取要嵌入的内容，并为每个 chunk 计算缓存键
        embedding_documents = [f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}" for chunk in bath_chunks]
        keys = [hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in embedding_documents]

        # 2. 只对未命中的 chunk 调用模型
        miss_indexes = [i for i, key in enumerate(keys) if key not in cache]
        if miss_indexes:
            miss_docs = [embedding_documents[i] for i in miss_indexes]
            try:
                embed_vector = embed_model.encode_documents(miss_docs)
            except Exception as e:
                raise EmbeddingError(message=f"嵌入失败,原因:{str(e)}", node_name=self.name)

            if not embed_vector:
                raise EmbeddingError(message="嵌入结果不存在")

            sparse_csr = embed_vector.get('sparse')
            for pos, chunk_index in enumerate(miss_indexes):
                cache[keys[chunk_index]] = {
                    "dense": embed_vector.get('dense')[pos].tolist(),
                    "sparse": self._extract_sparse_vector(sparse_csr, pos),
                }
            self.logger.info(f"嵌入缓存：命中 {len(bath_chunks) - len(miss_indexes)}，"
                             f"新计算 {len(miss_indexes)}")
        else:
            self.logger.info(f"嵌入缓存：{len(bath_chunks)} 个 chunk 全部命中")

        # 3. 回填向量到 chunk
        for i, chunk in enumerate(bath_chunks):
            cached = cache[keys[i]]
            chunk['dense_vector'] = cached["dense"]
            # JSON 落盘后稀疏键变成字符串，回填时还原为 int（Milvus 要求）
            chunk['sparse_vector'] = {int(k): v for k, v in cached["sparse"].items()}

        return bath_chunks

    @staticmethod
    def _cache_path(file_dir: str) -> str:
        if not file_dir:
            return ""
        return os.path.join(file_dir, _CACHE_FILE)

    def _load_cache(self, file_dir: str) -> Dict[str, Dict[str, Any]]:
        path = self._cache_path(file_dir)
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vectors = data.get("vectors") if isinstance(data, dict) else None
            if isinstance(vectors, dict):
                self.logger.info(f"加载嵌入缓存：{len(vectors)} 条")
                return vectors
        except Exception as e:
            self.logger.warning(f"嵌入缓存读取失败 {path}: {e}")
        return {}

    def _save_cache(self, file_dir: str, cache: Dict[str, Dict[str, Any]]) -> None:
        path = self._cache_path(file_dir)
        if not path:
            return
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "vectors": cache}, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            self.logger.warning(f"嵌入缓存写入失败 {path}: {e}")

    def _extract_sparse_vector(self, sparse_csr, index: int):
        """
        从稀疏矩阵中提取当前chunk对象的稀疏向量
        Args:
            sparse_csr:
            index:

        Returns:

        """
        # 3.1 从行索引中获取当前chunk的起始索引和结束索引
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]
        # 3.2 获取token_id
        token_id = sparse_csr.indices[start_index:end_index].tolist()
        # 3.3 获取weight
        weight = sparse_csr.data[start_index:end_index].tolist()

        # 3.4 返回单个chunk的稀疏向量值
        return dict(zip(token_id, weight))


if __name__ == '__main__':
    import json

    setup_logging()

    base_dir = Path(get_import_temp_dir())
    input_path = base_dir / "chunks_item_name.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data.get('chunks')})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")
