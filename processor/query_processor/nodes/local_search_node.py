from typing import Any, Dict

from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.processor.query_processor.nodes.hybrid_vector_search_node import HybridVectorSearch
from muti_rag.processor.query_processor.nodes.hyde_vector_search_node import HyDeVectorSearchNode


class LocalSearchNode(BaseNode):
    """本地召回组合节点：混合向量检索 →（按需）HyDE 假设文档检索。

    两步为什么合在一个节点里串行，而不是拆成图上的两个分支：
    HyDE 的触发判据（词汇覆盖率、是否空召回）依赖混合召回的结果，必须等它跑完；
    而 LangGraph 的汇合节点是"任一上游完成即触发"（实测：分支深度不等时 join 会被触发
    两次，下游 RRF/重排/答案跟着重复执行）。合成一个节点后，本地召回与联网检索在图上
    仍是同深度的并行分支，汇合语义保持正确。
    """

    name = "local_search_node"

    def __init__(self):
        super().__init__()
        self._hybrid = HybridVectorSearch()
        self._hyde = HyDeVectorSearchNode()

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(self._hybrid.process(state) or {})
        state.update(merged)
        merged.update(self._hyde.process(state) or {})
        return merged


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    node = LocalSearchNode()
    result = node.process({
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
        "sub_queries": ["RS-12 数字万用表如何测量直流电压？"],
        "query_plan": [{"use_hyde": "auto"}],
    })
    for i, hits in enumerate(result.get("embedding_chunks_per_q") or []):
        print(f"[混合 {i}] {len(hits)} 条")
    for i, hits in enumerate(result.get("hyde_embedding_chunks_per_q") or []):
        print(f"[HyDE {i}] {len(hits)} 条")
