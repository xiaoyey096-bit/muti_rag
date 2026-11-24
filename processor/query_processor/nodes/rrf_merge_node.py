from typing import Tuple, List, Dict, Any
from collections.abc import Mapping
from muti_rag.processor.query_processor.base import BaseNode, T
from muti_rag.processor.query_processor.state import QueryGraphState


class RrfMergeNode(BaseNode):

    name = "rrf_merge_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 逐个子问题独立做 RRF 融合（每问各自的两路结果，不跨问混排）
        sub_queries = state.get('sub_queries') or [state.get('rewritten_query') or ""]
        emb_per_q = state.get('embedding_chunks_per_q') or []
        hyde_per_q = state.get('hyde_embedding_chunks_per_q') or []

        per_q: List[List[Dict[str, Any]]] = []
        for i in range(len(sub_queries)):
            embedding_chunks = emb_per_q[i] if i < len(emb_per_q) else []
            hyde_embedding_chunks = hyde_per_q[i] if i < len(hyde_per_q) else []

            # 2. 两路等权（rrf 得分影响因子最大的是 k 而非 weight）
            rrf_inputs = [
                (self._validate_search_result(embedding_chunks), 1.0),
                (self._validate_search_result(hyde_embedding_chunks), 1.0),
            ]

            # 3. RRF 计算 + 把得分写回 chunk 便于观测
            merged_rrf = []
            for rrf_res, rrf_score in self._merge_rrf_docs(rrf_inputs,
                                                           self.config.rrf_k,
                                                           self.config.rrf_max_results):
                rrf_res['rrf_score'] = rrf_score
                merged_rrf.append(rrf_res)
            per_q.append(merged_rrf)

        # 4. 更新state（per_q 为权威；legacy 保留首问结果）
        state['rrf_chunks_per_q'] = per_q
        state['rrf_chunks'] = per_q[0] if per_q else []

        return state

    def _merge_rrf_docs(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], rrf_k: int, rrf_max_results: int) -> \
            List[Tuple[Dict[str, Any], float]]:
        """
        RRF计算多路检索返回的文档得分
        公式：weight(i)/ k+rank(i)[doc]
        Args:
            rrf_inputs:  多路检索的文档以及对应的权重
            rrf_k:  平滑参数
            rrf_max_results: 最大返回的个数

        Returns:
           多路检索文档对象以及经过RRF计算之后的文档得分
           Tuple[Dict,Float]:第一个元素是文档对象 第二个元素是文档对象对应的得分

        """

        chunk_score = {}
        chunk_data = {}  # chunk_id 和chunk对象

        # 1. 遍历所有路的检索结果
        for search_result, weight in rrf_inputs:

            # 2. 遍历某一路的检索结果
            for rank, entity in enumerate(search_result, 1):

                # 2.1 获取chunk_id
                chunk_id = entity.get('chunk_id')

                # 2.2 判断chunk_id
                if not chunk_id:
                    continue

                # 2.3 存储chunk_id的分数
                chunk_score[chunk_id] = chunk_score.get(chunk_id, float(0)) + weight / (rrf_k + rank)

                # 2.4 存储chunk_id 和chunk对象
                chunk_data.setdefault(chunk_id, entity)  # （默认根据key）去重

        # 3. 排序以及构建chunk对象和得分的结果
        final_rrf_result = sorted([(chunk_data.get(chunk_id), score) for chunk_id, score in chunk_score.items()],
                                  key=lambda x: x[1], reverse=True)

        # 4. 返回(简单使用rrf_max_results判断 返回)
        return final_rrf_result[:rrf_max_results] if rrf_max_results else final_rrf_result

    def _validate_search_result(self, search_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        校验多路检索的结果
        Args:
            search_result:

        Returns:
            字典列表[字典对象中只有entity字段的内容]
        """

        # 1. 判断是否为空
        if not search_result:
            return []

        # 2. 遍历字典对象
        search_results = []
        for res in search_result:

            # 2.1 判断对象是否存在以及对应类型是否预期
            if not res or not isinstance(res, Mapping):
                continue

            # 2.2 获取 entity对象
            entity = res.get('entity')

            # 2.3 判断对象是否存在以及对应类型是否预期
            if not entity or not isinstance(entity, dict):
                continue

            search_results.append(entity)

        return search_results


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfMergeNode()
    result = rrf_node.process(mock_state)
