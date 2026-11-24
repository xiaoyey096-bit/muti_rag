from typing import Tuple, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.processor.query_processor.exceptions import StateFieldError
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.client.storage_clients import StorageClients
from muti_rag.utils.embedding_util import generate_bge_m3_hybrid_vectors
from muti_rag.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, build_search_filter


class HybridVectorSearch(BaseNode):
    """混合（稠密+稀疏）向量检索。

    V5 改动：所有子问题一次批量嵌入（BGE 前向按批只跑一次），检索按子问题并发发出。
    子问数增加时，本地检索部分不再线性变慢。
    """

    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        rewritten_query, item_names = self._validate_state(state)
        sub_queries = state.get('sub_queries') or [rewritten_query]

        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return {"embedding_chunks_per_q": [[] for _ in sub_queries], "embedding_chunks": []}

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return {"embedding_chunks_per_q": [[] for _ in sub_queries], "embedding_chunks": []}

        expr, expr_params = build_search_filter(item_names, state.get('kb_id', ''))
        limit = self.config.embedding_search_limit

        # 1. 批量嵌入：一次前向拿全部子问题的混合向量
        try:
            vectors = generate_bge_m3_hybrid_vectors(model=bge_m3_client, embedding_documents=list(sub_queries))
        except Exception as e:
            self.logger.error(f"子问题批量向量化失败 原因:{str(e)}")
            return {"embedding_chunks_per_q": [[] for _ in sub_queries], "embedding_chunks": []}

        # 2. 每问独立召回（各自 top_k，不共用名额）
        def search_one(index: int) -> List[Any]:
            try:
                requests = create_hybrid_search_requests(dense_vector=vectors['dense'][index],
                                                         sparse_vector=vectors['sparse'][index],
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
                return res[0] if res and res[0] else []
            except Exception as e:
                self.logger.error(f"子问题{sub_queries[index]}执行混合搜索查询失败 原因:{str(e)}")
                return []

        if len(sub_queries) == 1:
            per_q = [search_one(0)]
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as pool:
                per_q = list(pool.map(search_one, range(len(sub_queries))))

        return {"embedding_chunks_per_q": per_q,
                "embedding_chunks": per_q[0] if per_q else []}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names


if __name__ == '__main__':
    import json

    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"],
        "sub_queries": ["万用表如何测量电阻", "万用表如何测量电压"],
    }

    vector_search = HybridVectorSearch()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks_per_q'):
        print(json.dumps(r, ensure_ascii=False, indent=2)[:600])
