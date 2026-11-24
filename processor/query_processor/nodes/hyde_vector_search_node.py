import time
from typing import Tuple, List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import SystemMessage, HumanMessage
from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.processor.query_processor.exceptions import StateFieldError
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.client.storage_clients import StorageClients
from muti_rag.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from muti_rag.utils.embedding_util import generate_bge_m3_hybrid_vectors
from muti_rag.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, build_search_filter
from muti_rag.utils.evidence_util import lexical_coverage

# 子问题级 HyDE 假设文档缓存：{(kb_id, sub_query): (时间戳, 文档)}
_HYDE_CACHE: Dict[Tuple[str, str], Tuple[float, str]] = {}


class HyDeVectorSearchNode(BaseNode):
    """HyDE 假设文档检索（V5：按需触发，不再必经）。

    触发条件：
    - 查询理解标注 use_hyde=yes（表述模糊/概念类问题）；
    - 或标注 auto 且混合召回的词汇覆盖率低于阈值——问题里的关键词没在候选切片里
      出现，说明纯字面/向量召回定位不到答案，这时假设文档才有增量价值；
    - 或该子问题混合召回为空。
    明确查值且召回已经覆盖问题词表（auto 判定通过）时直接跳过，省掉一次远程 LLM 调用。
    """

    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        rewritten_query, item_names = self._validate_state(state)
        sub_queries = state.get('sub_queries') or [rewritten_query]
        empty = {"hyde_embedding_chunks_per_q": [[] for _ in sub_queries], "hyde_embedding_chunks": []}

        if not self.config.hyde_enabled:
            self.logger.info("HYDE_ENABLED=false，跳过 HyDE 分支")
            return empty

        # 1. 决定哪些子问题需要 HyDE
        todo = self._select_sub_queries(state, sub_queries, item_names)
        if not todo:
            self.logger.info("HyDE 判定：全部子问本地召回覆盖足够，跳过假设文档")
            return empty

        # 2. 并行生成假设文档（远程 LLM 调用，彼此独立）
        docs = self._generate_hy_documents(todo, sub_queries, item_names, state.get('kb_id', ''))

        # 3. 批量嵌入 + 并发检索
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"HyDE 依赖客户端获取失败 原因:{str(e)}")
            return empty

        pairs = [f"{sub_queries[i]}\n{docs[i]}" for i in todo if docs.get(i)]
        if not pairs:
            return empty
        try:
            vectors = generate_bge_m3_hybrid_vectors(bge_m3_client, pairs)
        except Exception as e:
            self.logger.error(f"假设文档向量化失败 原因:{str(e)}")
            return empty

        # 向量下标与子问题下标一一对应（pairs 与 todo 顺序一致）
        work = [(vec_index, sub_index) for vec_index, sub_index in
                zip(range(len(pairs)), [i for i in todo if docs.get(i)])]

        expr, expr_params = build_search_filter(item_names, state.get('kb_id', ''))
        limit = self.config.hyde_search_limit
        per_q: List[List[Any]] = [[] for _ in sub_queries]

        def search_one(vec_index: int, sub_index: int) -> Tuple[int, List[Any]]:
            try:
                requests = create_hybrid_search_requests(dense_vector=vectors['dense'][vec_index],
                                                         sparse_vector=vectors['sparse'][vec_index],
                                                         expr=expr,
                                                         expr_params=expr_params,
                                                         limit=limit)
                res = execute_hybrid_search_query(milvus_client=milvus_client,
                                                 collection_name=self.config.chunks_collection,
                                                 search_requests=requests,
                                                 ranker_weights=(0.5, 0.5),
                                                 norm_score=True,
                                                 limit=limit,
                                                 output_fields=["chunk_id", "content", "item_name", 'title'])
                return sub_index, (res[0] if res and res[0] else [])
            except Exception as e:
                self.logger.error(f"子问题{sub_queries[sub_index]}假设文档混合检索失败 原因:{str(e)}")
                return sub_index, []

        if len(work) == 1:
            idx, hits = search_one(*work[0])
            per_q[idx] = hits
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(work))) as pool:
                for idx, hits in pool.map(lambda w: search_one(*w), work):
                    per_q[idx] = hits

        self.logger.info(f"HyDE 触发子问 {len(work)}/{len(sub_queries)} 个")
        return {"hyde_embedding_chunks_per_q": per_q,
                "hyde_embedding_chunks": per_q[0] if per_q else []}

    def _select_sub_queries(self, state: QueryGraphState, sub_queries: List[str],
                            item_names: List[str]) -> List[int]:
        """按标注 + 混合召回质量挑出需要 HyDE 的子问题下标。"""
        plan = state.get('query_plan') or []
        emb_per_q = state.get('embedding_chunks_per_q') or []
        todo: List[int] = []
        for i, sub in enumerate(sub_queries):
            meta = plan[i] if i < len(plan) else {}
            flag = meta.get('use_hyde', 'auto')
            hits = emb_per_q[i] if i < len(emb_per_q) else []
            if flag == 'no':
                continue
            if flag == 'yes':
                todo.append(i)
                continue
            contents = [((h.get('entity') or {}).get('content') or '') for h in hits]
            coverage = lexical_coverage(sub, contents, strip_terms=item_names)
            if not hits or coverage < self.config.hyde_coverage_floor:
                self.logger.info(f"子问「{sub}」混合召回覆盖率{coverage:.2f}<{self.config.hyde_coverage_floor}，补 HyDE")
                todo.append(i)
        return todo

    def _generate_hy_documents(self, todo: List[int], sub_queries: List[str],
                               item_names: List[str], kb_id: str) -> Dict[int, Optional[str]]:
        """并行生成假设文档，命中缓存的直接复用。"""
        docs: Dict[int, Optional[str]] = {}
        pending: List[int] = []
        now = time.time()
        for i in todo:
            key = (kb_id, sub_queries[i])
            cached = _HYDE_CACHE.get(key)
            if cached and now - cached[0] < self.config.hyde_cache_ttl_s:
                docs[i] = cached[1]
            else:
                docs[i] = None
                pending.append(i)

        if not pending:
            self.logger.info(f"HyDE 全部命中缓存（{len(todo)} 个子问）")
            return docs

        with ThreadPoolExecutor(max_workers=min(4, len(pending))) as pool:
            results = list(pool.map(lambda i: self._generate_hy_document(sub_queries[i], item_names), pending))
        for i, doc in zip(pending, results):
            docs[i] = doc
            if doc:
                _HYDE_CACHE[(kb_id, sub_queries[i])] = (time.time(), doc)
        return docs

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names

    def _generate_hy_document(self, sub_query: str, item_names: List[str]) -> Optional[str]:
        """为单个子问题生成假设性文档（HyDE）。"""
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"获取LLM客户端失败 原因:{str(e)}")
            return None

        system_prompt = f"您是一位{item_names}方面的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names=item_names,
                                                        rewritten_query=sub_query,
                                                        max_chars=self.config.hyde_doc_chars)
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"LLM生成{item_names}的假设性文档失败 原因:{str(e)}")
            return None

        content = (getattr(llm_response, 'content', '') or '').strip()
        return content or None


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
        "sub_queries": ["RS-12 数字万用表如何测量直流电压？"],
        "query_plan": [{"use_hyde": "yes"}],
        "embedding_chunks_per_q": [[]],
    }

    node = HyDeVectorSearchNode()
    result = node.process(mock_state)
    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} title={entity.get('title')} "
              f"distance={chunk.get('distance', 'N/A')}")
