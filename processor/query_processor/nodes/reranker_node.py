from typing import Tuple, List, Dict, Any
import math
import logging
from FlagEmbedding import FlagReranker
from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.image_ref_util import doc_image_urls, parse_image_refs
from muti_rag.utils.evidence_util import lexical_coverage

logger = logging.getLogger(__name__)


def batch_score_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
    """一次 compute_score 打完所有 (问题, 文本) 对；模型不可用时返回 None 列表。"""
    if not pairs:
        return []
    try:
        rerank_client: FlagReranker = AIClients.get_bge_m3_rerank_client()
    except ConnectionError as e:
        logger.error(f"获取BGE-M3重排序模型失败 原因:{str(e)}")
        return [None] * len(pairs)
    try:
        raw = rerank_client.compute_score(sentence_pairs=pairs)
    except Exception as e:
        logger.error(f"BGE-M3重排序模型计算分数失败 原因:{str(e)}")
        return [None] * len(pairs)
    if not isinstance(raw, list):
        raw = [raw]
    return [1.0 / (1.0 + math.exp(-float(s))) if s is not None else None for s in raw]


class RerankerNode(BaseNode):
    """精排 + 动态 top_k + 证据统计（V5）。

    三点变化：
    1. 所有子问题的 (问题, 切片) 与 (子问题, 图题) 合并成一次 compute_score 调用，
       重排开销不再随子问数线性增长。
    2. 重排绝对分不可信（50 题标定：库外题 top1 中位 0.96、可答题 0.013，方向相反），
       因此按召回顺序保底：RRF 前 local_recall_keep 的本地切片一定进入上下文，
       避免断崖把唯一含答案的切片切掉（D03 就是这么错的）。
    3. 输出每问证据统计 evidence_per_q，交给证据门控决定 本地/联网/固定模板。
    """

    name = "reranker_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        sub_queries = state.get('sub_queries') or [state.get('rewritten_query') or state.get('original_query')]
        plan = state.get('query_plan') or []
        rrf_per_q = state.get('rrf_chunks_per_q') or []
        web_per_q = state.get('web_search_docs_per_q') or []

        # 1. 逐子问收集候选（图片意图只看本地含图切片）
        candidates: List[List[Dict[str, Any]]] = []
        image_modes: List[str] = []
        for i, sub in enumerate(sub_queries):
            rrf_chunks = rrf_per_q[i] if i < len(rrf_per_q) else (state.get('rrf_chunks') or [] if i == 0 else [])
            web_docs = web_per_q[i] if i < len(web_per_q) else (state.get('web_search_docs') or [] if i == 0 else [])
            docs = self._collect_rerank_inputs(rrf_chunks, web_docs)
            mode = (plan[i] if i < len(plan) else {}).get('image_mode', 'none')
            if mode == 'show':
                docs = [d for d in docs if d.get('source') == 'local' and doc_image_urls(d)]
            for d in docs:
                d['image_refs'] = parse_image_refs(d.get('content', ''))
            candidates.append(docs)
            image_modes.append(mode)

        # 2. 一次批量重排：全部切片对 + 全部图题对
        doc_pairs, caption_pairs = self._build_pairs(sub_queries, candidates, image_modes)
        scores = self._batch_score(doc_pairs + caption_pairs)
        doc_scores = scores[:len(doc_pairs)]
        caption_scores = scores[len(doc_pairs):]

        # 3. 图题分数写回候选切片
        self._attach_caption_scores(candidates, image_modes, caption_scores)

        # 4. 逐子问截断 + 证据统计
        per_q: List[List[Dict[str, Any]]] = []
        evidence: List[Dict[str, Any]] = []
        offset = 0
        for i, sub in enumerate(sub_queries):
            docs = candidates[i]
            scored = self._apply_scores(docs, doc_scores[offset:offset + len(docs)])
            offset += len(docs)

            local_recall = [d for d in docs if d.get('source') == 'local']
            web_recall = [d for d in docs if d.get('source') == 'web']

            if image_modes[i] == 'show':
                kept = self._image_cutoff(scored)
                self._select_images(kept, sub)
            else:
                kept = self._cliff_cutoff(scored, local_recall)
                if image_modes[i] == 'read':
                    self._select_images(kept, sub)

            per_q.append(kept)
            evidence.append({
                "sub": sub,
                "image_mode": image_modes[i],
                "local_recall": len(local_recall),
                "web_recall": len(web_recall),
                "kept_local": sum(1 for d in kept if d.get('source') == 'local'),
                "kept_web": sum(1 for d in kept if d.get('source') == 'web'),
                "best_score": max([d['score'] for d in scored if d.get('score') is not None], default=None),
                "coverage": round(lexical_coverage(sub, [d.get('content', '') for d in local_recall],
                                                   strip_terms=state.get('item_names') or []), 4),
            })

        state['reranked_docs_per_q'] = per_q
        state['reranked_docs'] = per_q[0] if per_q else []
        state['evidence_per_q'] = evidence
        for i, e in enumerate(evidence):
            self.logger.info(f"证据[{i}] kept本地{e['kept_local']}/联网{e['kept_web']} "
                             f"召回{e['local_recall']}/{e['web_recall']} 词覆盖{e['coverage']} "
                             f"最强重排分={e['best_score']}")
        return state

    # ==================== 候选收集 ====================

    def _collect_rerank_inputs(self, rrf_chunks: List[Dict[str, Any]],
                               web_search_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化单个子问题的本地+联网检索结果。"""
        final_docs = []
        for chunk in rrf_chunks or []:
            if not chunk or not isinstance(chunk, dict):
                continue
            content = chunk.get('content', '')
            if not content:
                continue
            final_docs.append(self._format_doc(content=content,
                                               chunk_id=chunk.get('chunk_id'),
                                               title=chunk.get('title', ''),
                                               source="local",
                                               rrf_score=chunk.get('rrf_score')))

        for doc in web_search_docs or []:
            if not doc or not isinstance(doc, dict):
                continue
            content = doc.get('snippet', '')
            if not content:
                continue
            final_docs.append(self._format_doc(content=content,
                                               title=doc.get('title', ''),
                                               url=doc.get('url', ''),
                                               source="web"))

        self.logger.info(f"获取Reranker阶段需要的搜索结果个数{len(final_docs)}")
        return final_docs

    def _format_doc(self, content: str, chunk_id: int = None, title: str = "", url: str = "", source: str = "",
                    rrf_score: float = None):
        return {
            "content": content,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "source": source,
            "rrf_score": rrf_score
        }

    # ==================== 批量打分 ====================

    def _build_pairs(self, sub_queries: List[str], candidates: List[List[Dict[str, Any]]],
                     image_modes: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """把全部子问的切片对与图题对拼成一个批量，顺序与写回时一致。"""
        doc_pairs: List[Tuple[str, str]] = []
        caption_pairs: List[Tuple[str, str]] = []
        for i, docs in enumerate(candidates):
            for d in docs:
                doc_pairs.append((sub_queries[i], d.get('content')))
            if image_modes[i] in ('show', 'read'):
                for d in docs:
                    for ref in d['image_refs']:
                        caption_pairs.append((sub_queries[i], ref['caption'] or d.get('title', '') or ref['url']))
        return doc_pairs, caption_pairs

    def _attach_caption_scores(self, candidates: List[List[Dict[str, Any]]], image_modes: List[str],
                               caption_scores: List[float]):
        """按 _build_pairs 的顺序把图题分写回 doc['image_ref_scores']。"""
        k = 0
        for i, docs in enumerate(candidates):
            if image_modes[i] not in ('show', 'read'):
                continue
            for d in docs:
                if not d['image_refs']:
                    continue
                d['image_ref_scores'] = caption_scores[k:k + len(d['image_refs'])]
                k += len(d['image_refs'])

    def _batch_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        return batch_score_pairs(pairs)

    @staticmethod
    def _sigmoid(score: float) -> float:
        """sigmoid归一化，将 (-∞, +∞) 映射到 (0, 1)"""
        return 1.0 / (1.0 + math.exp(-score))

    def _apply_scores(self, docs: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        scored = [{**d, "score": s} for d, s in zip(docs, scores)]
        return sorted(scored, key=lambda x: (x['score'] is not None, x['score'] or 0), reverse=True)

    # ==================== 截断策略 ====================

    def _cliff_cutoff(self, refine_docs: List[Dict[str, Any]],
                      local_recall_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """动态 top_k + 本地优先 + 召回顺序保底。

        先用断崖阈值决定保留多少个（min_top_k 兜底），再保证前 min_top_k 个本地切片
        一定入选，然后把 RRF 排名前 local_recall_keep 的本地切片补进来——重排分不可靠，
        召回顺序是独立于它的第二证据。剩余名额按分数补，总数不超过 max_top_k。
        """
        if not refine_docs:
            return []
        max_top_k = self.config.rerank_max_top_k
        rerank_min_top_k = self.config.rerank_min_top_k
        upper_bound = min(max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        cut_off = upper_bound
        max_gap = 0

        for i in range(0, upper_bound - 1):
            current_score = refine_docs[i].get('score')
            next_score = refine_docs[i + 1].get('score')
            if current_score is None or next_score is None:
                continue
            gap = current_score - next_score
            if gap >= 0.15 and gap > max_gap:
                max_gap = gap
                cut_off = i + 1

        cut_off = max(cut_off, lower_bound)

        local_docs = [d for d in refine_docs if d.get('source') == 'local']
        kept = local_docs[:rerank_min_top_k]
        kept_ids = {id(d) for d in kept}

        # 召回顺序保底：RRF 前 N 个本地切片必然进入上下文
        for d in local_recall_docs[:self.config.local_recall_keep]:
            match = next((x for x in refine_docs if x.get('chunk_id') == d.get('chunk_id')), None)
            if match and id(match) not in kept_ids:
                match['recall_backup'] = True
                kept.append(match)
                kept_ids.add(id(match))

        for d in refine_docs:
            if len(kept) >= cut_off:
                break
            if id(d) not in kept_ids:
                kept.append(d)
                kept_ids.add(id(d))

        return kept[:max_top_k]

    def _image_cutoff(self, refine_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """图片意图的动态 top_k：相关性门槛 + 断崖，不做本地保底。

        普通问答的"本地保底 min_top_k"会把无关本地切片（如 Dashboard 截图）强塞进
        top3，导致问接口截面图却回两张图。图片意图下只保留重排分过门槛且断崖内的切片；
        一张都没过强门槛时，保留分数最高的一篇作为低置信候选（避免"有图说成没图"）。
        """
        if not refine_docs:
            return []
        if all(d.get('score') is None for d in refine_docs):
            return refine_docs[:self.config.rerank_min_top_k]

        floor = self.config.image_caption_floor
        kept = [d for d in refine_docs if (d.get('score') or 0) >= floor][:self.config.rerank_max_top_k]
        if kept:
            cut_off, max_gap = len(kept), 0
            for i in range(len(kept) - 1):
                gap = kept[i]['score'] - kept[i + 1]['score']
                if gap >= 0.15 and gap > max_gap:
                    max_gap = gap
                    cut_off = i + 1
            return kept[:cut_off]

        best = refine_docs[0]
        if (best.get('score') or 0) >= self.config.image_caption_weak_floor:
            best['low_confidence'] = True
            self.logger.info(f"配图切片未过强门槛，保留低置信候选：{str(best.get('title'))[:30]}")
            return [best]
        return []

    def _select_images(self, kept_docs: List[Dict[str, Any]], sub_query: str):
        """图题级相关性过滤：同一章节切片可能含多张图，只保留图题与问题相关者。"""
        strong, weak = [], []
        for doc in kept_docs:
            doc['matched_images'] = []
            scores = doc.get('image_ref_scores') or []
            for pos, ref in enumerate(doc.get('image_refs') or []):
                score = scores[pos] if pos < len(scores) else None
                if score is None:
                    weak.append((doc, ref, 1.0))  # 打分失败：不作硬过滤，按低置信保留
                elif score >= self.config.image_caption_floor:
                    strong.append((doc, ref, score))
                elif score >= self.config.image_caption_weak_floor:
                    weak.append((doc, ref, score))

        limit = self.config.image_max_show
        for doc, ref, _ in strong[:limit]:
            doc['matched_images'].append(ref)

        if not strong and weak:
            weak.sort(key=lambda x: x[2], reverse=True)
            doc, ref, score = weak[0]
            doc['matched_images'].append(ref)
            doc['low_confidence'] = True
            self.logger.info(f"图题无强相关，保留低置信配图 1 张（{score:.3f}）")

        total = sum(len(d.get('matched_images') or []) for d in kept_docs)
        self.logger.info(f"图片意图图题过滤：候选{len(strong) + len(weak)}张，保留{total}张")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    node = RerankerNode()
    result = node.process(mock_state)
    print(f"reranked: {len(result.get('reranked_docs') or [])}")
    print(f"evidence: {result.get('evidence_per_q')}")
