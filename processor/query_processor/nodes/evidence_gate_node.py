from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.processor.query_processor.nodes.reranker_node import batch_score_pairs
from muti_rag.processor.query_processor.nodes.web_mcp_search_node import web_search_once

MODE_LOCAL = "local"
MODE_WEB = "web"
MODE_REFUSAL = "refusal"


class EvidenceGateNode(BaseNode):
    """证据门控：逐子问决定"用什么证据回答"，并在需要时按需补联网。

    判定顺序（子问级）：
    1. 保留了本地切片 → local；
    2. 只保留了联网切片 → web；
    3. 什么都没保留：
       - 图片意图（show）→ refusal（由答案节点给"暂无配图"模板，不编造）；
       - 开了联网且未被标注为"不需要联网" → 此时才补一次联网检索，
         有结果转 web，没有转 refusal；
       - 其余 → refusal。
    这里也顺带完成"联网只在本地证据不足时才发生"的收口：查询理解标注 auto 的子问
    不会在联网分支里提前调用外部搜索，省掉绝大多数无用的外部请求。

    注意：本节点只处理"检索层完全没有证据"的硬门。文档里确实没有该信息但向量库
    仍然返回了 top_k 噪音切片的情况（50 题标定显示库外题 top1 余弦 0.56~0.76、重排
    0.89~0.999，与可答题完全重叠，无法用分数阈值区分）交给答案节点的无证据契约处理：
    模型判无可依据时输出 NO_EVIDENCE，由答案节点换成固定模板或转联网重答。
    """

    name = "evidence_gate_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        sub_queries = state.get('sub_queries') or [state.get('rewritten_query') or ""]
        plan = state.get('query_plan') or []
        evidence = state.get('evidence_per_q') or []
        per_docs = state.get('reranked_docs_per_q') or []
        web_per_q = state.get('web_search_docs_per_q') or []
        enable_web = bool(state.get('enable_web', True))
        item_names = state.get('item_names') or []

        modes: List[str] = []
        deferred: List[int] = []
        for i, sub in enumerate(sub_queries):
            ev = evidence[i] if i < len(evidence) else {}
            meta = plan[i] if i < len(plan) else {}
            docs = per_docs[i] if i < len(per_docs) else []

            if any(d.get('source') == 'local' for d in docs):
                modes.append(MODE_LOCAL)
            elif any(d.get('source') == 'web' for d in docs):
                modes.append(MODE_WEB)
            elif meta.get('image_mode') == 'show':
                modes.append(MODE_REFUSAL)
            elif enable_web and meta.get('needs_web', 'auto') != 'no' and not (web_per_q[i] if i < len(web_per_q) else []):
                deferred.append(i)
                modes.append(MODE_REFUSAL)  # 补检索后再改写
            else:
                modes.append(MODE_REFUSAL)

        if deferred:
            self.logger.info(f"本地无证据且允许联网，按需补检索 {len(deferred)} 个子问")
            with ThreadPoolExecutor(max_workers=min(4, len(deferred))) as pool:
                results = list(pool.map(lambda i: web_search_once(sub_queries[i], item_names), deferred))
            for i, docs in zip(deferred, results):
                if not docs:
                    continue
                scored = self._to_docs(sub_queries[i], docs)
                bucket = per_docs[i] if i < len(per_docs) else []
                if i < len(per_docs):
                    per_docs[i] = bucket + scored
                else:
                    per_docs.append(scored)
                web_bucket = web_per_q[i] if i < len(web_per_q) else []
                if i < len(web_per_q):
                    web_per_q[i] = web_bucket + docs
                modes[i] = MODE_WEB

        state['reranked_docs_per_q'] = per_docs
        state['web_search_docs_per_q'] = web_per_q
        state['answer_mode_per_q'] = modes
        state['answer_mode'] = "mixed" if len(set(modes)) > 1 else (modes[0] if modes else MODE_REFUSAL)
        self.logger.info(f"证据门控结果: {modes}")
        return state

    def _to_docs(self, sub_query: str, web_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把补检索到的网页片段转成与重排节点同构的文档，并批量打分供截断使用。"""
        pairs = [(sub_query, d.get('snippet', '')) for d in web_docs if d.get('snippet')]
        scores = batch_score_pairs(pairs)
        out = []
        for doc, score in zip(web_docs, scores):
            if not doc.get('snippet'):
                continue
            out.append({"content": doc['snippet'], "chunk_id": None, "title": doc.get('title', ''),
                        "url": doc.get('url', ''), "source": "web", "rrf_score": None,
                        "score": score, "deferred_web": True})
        return out


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    node = EvidenceGateNode()
    mock = {
        "sub_queries": ["DGX Spark 的保修期是几年？"],
        "query_plan": [{"needs_web": "auto", "image_mode": "none"}],
        "evidence_per_q": [{"kept_local": 0, "kept_web": 0, "local_recall": 5, "web_recall": 0}],
        "reranked_docs_per_q": [[]],
        "web_search_docs_per_q": [[]],
        "enable_web": False,
        "item_names": ["DGX Spark"],
    }
    print(node.process(mock)['answer_mode_per_q'])
