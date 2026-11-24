import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.mongo_history_util import save_chat_message
from muti_rag.utils.task_util import set_task_result
from muti_rag.utils.sse_util import push_sse_event, SSEEvent
from muti_rag.prompts.query_prompt import (ANSWER_PROMPT, REFUSAL_TEMPLATE, IMAGE_NONE_TEMPLATE,
                                           WEB_SOURCE_NOTICE, WEB_SOURCE_NOTICE_MIXED)
from muti_rag.utils.image_ref_util import doc_image_urls, parse_image_refs, norm_img_url
from muti_rag.utils.vl_answer_util import answer_with_images
from muti_rag.utils.evidence_util import is_no_evidence_answer
from muti_rag.processor.query_processor.nodes.web_mcp_search_node import web_search_once
from muti_rag.processor.query_processor.nodes.reranker_node import batch_score_pairs


class AnswerOutPutNode(BaseNode):
    """答案生成 + 证据分级出口（V5）。

    1. 无证据不再让模型自由发挥：证据门控判 refusal 的子问直接出固定模板（零次 LLM 调用）；
       检索有结果但撑不起答案的，走提示词里的 NO_EVIDENCE 契约，命中后同样换模板。
    2. 只用联网证据/混合证据回答时，末尾追加来源提示（模板固定，不由模型生成）。
    3. 上下文工程：多子问按子问数均分总预算（有下限），并做跨子问切片去重，
       避免同一个 chunk 在多个子问里重复占位、挤掉真正互补的证据；去重让本问判无证据时，
       先把兄弟子问的证据并回来重试一次（同库同文档，不该因为拆分位置而拒答），再退联网/模板。
    4. 非流式的多子问并行生成，按子问顺序合并；流式保持顺序输出（SSE 不能乱序）。
    """

    name = "answer_output_node"

    _IMAGE_DOC_TOP_N = 3
    _IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)]+)\)")
    _IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|bmp|webp)(\?|#|$)", re.I)
    _FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
    _EMPTY_HEADINGS = {"# 相关配图", "# 参考链接"}
    _NO_EVIDENCE_PREFIXES = ["N", "NO", "NO_", "NO_E", "NO_EV", "NO_EVI", "NO_EVID", "NO_EVIDE",
                             "NO_EVIDENC", "NO_EVIDENCE"]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        is_stream = state.get('is_stream')
        task_id = state.get('task_id')

        if state.get('answer'):
            # 澄清/短路分支：已有答案，直接推送
            is_streamed = False
        else:
            sub_queries = state.get('sub_queries') or [state.get('rewritten_query') or state.get('original_query') or ""]
            per_docs = self._dedup_docs(state, sub_queries)
            plan = state.get('query_plan') or []
            modes = state.get('answer_mode_per_q') or []
            multi = len(sub_queries) > 1
            budget = self._sub_budget(len(sub_queries))

            jobs = []
            for i, sub in enumerate(sub_queries):
                docs = per_docs[i] if i < len(per_docs) else []
                meta = plan[i] if i < len(plan) else {}
                mode = modes[i] if i < len(modes) else ("local" if docs else "refusal")
                prefix = self._section_prefix(sub, i, multi)
                siblings = [d for j, other in enumerate(per_docs) if j != i for d in other]
                jobs.append((i, sub, docs, meta, mode, prefix, siblings))

            if multi and not is_stream:
                with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                    parts = list(pool.map(lambda j: self._answer_job(j, state), jobs))
            else:
                parts = [self._answer_job(j, state) for j in jobs]

            state['answer'] = "".join(parts)
            is_streamed = is_stream

        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get('answer'))

        self.save_history(state)

        if is_stream:
            push_sse_event(task_id=task_id, event=SSEEvent.FINAL,
                           data={"answer": state.get('answer')})

        return state

    # ==================== 子问级生成 ====================

    def _answer_job(self, job: Tuple[int, str, List[Dict[str, Any]], Dict[str, Any], str, str, List[Dict[str, Any]]],
                    state: QueryGraphState) -> str:
        _, sub, docs, meta, mode, prefix, siblings = job
        image_mode = meta.get('image_mode', 'none')
        if mode == 'refusal':
            self.logger.info(f"子问「{sub[:30]}」无可用证据，走固定模板（不调用 LLM）")
            body = IMAGE_NONE_TEMPLATE if image_mode == 'show' else REFUSAL_TEMPLATE
        elif image_mode in ('show', 'read'):
            body = self._image_answer(sub, docs, state, image_mode)
        else:
            body = self._text_answer(sub, docs, state, has_web=self._has_web(docs),
                                     has_local=self._has_local(docs), siblings=siblings)
        return prefix + self._scrub_internals(body)

    _META_LINE_RE = re.compile(r"\[文档[:：]\d+\][^\n]*")
    _META_TOKEN_RE = re.compile(r"\[?\s*(?:chunk_id|doc_id|score|source)\s*[:=]\s*[^\s\]]+\s*\]?")
    _EMPTY_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n\s*```")
    _EMPTY_ITEM_RE = re.compile(r"^\s*[-*+]\s*$", re.M)
    _HEADING_SPLIT_RE = re.compile(r"(?m)^(#{1,6}\s.*)$")

    def _scrub_internals(self, text: str) -> str:
        """输出卫生：模型偶尔把参考内容的内部标记（[文档:N]/chunk_id/score）抄进答案，
        抄完还会留下只有标记的空壳小节。这里统一剔除，并删掉被清空的列表项与标题段。"""
        # 系统追加的来源提示先摘下来，避免它让被清空的"## 依据来源"看起来还有正文
        text, tail = self._split_notice(text)
        cleaned = self._META_TOKEN_RE.sub("", self._META_LINE_RE.sub("", text))
        cleaned = self._EMPTY_ITEM_RE.sub("", self._EMPTY_FENCE_RE.sub("", cleaned))
        parts = self._HEADING_SPLIT_RE.split(cleaned)
        out = [parts[0]]
        for i in range(1, len(parts), 2):
            body = parts[i + 1] if i + 1 < len(parts) else ""
            if body.strip():
                out.append(parts[i])
                out.append(body)
        return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip() + tail

    @staticmethod
    def _split_notice(text: str) -> Tuple[str, str]:
        for notice in (WEB_SOURCE_NOTICE, WEB_SOURCE_NOTICE_MIXED):
            if text.endswith(notice):
                return text[:-len(notice)], notice
        return text, ""

    def _text_answer(self, sub: str, docs: List[Dict[str, Any]], state: QueryGraphState,
                     has_web: bool, has_local: bool,
                     siblings: Optional[List[Dict[str, Any]]] = None) -> str:
        body = self._generate(self._build_prompt(sub, docs, state), sub, state)
        body = self._sanitize_images(body, docs)

        if not is_no_evidence_answer(body):
            return body + self._source_notice(has_local, has_web)

        # 模型判"参考内容撑不起答案"：先看同一次提问里兄弟子问的证据（同一份文档、
        # 只是被拆分/去重错开），再联网，都不行才固定模板。
        self.logger.info(f"子问「{sub[:30]}」模型判无证据，转兄弟子问证据/联网/固定模板")
        if siblings:
            merged = self._merge_docs(docs, siblings)
            retry = self._sanitize_images(self._generate(self._build_prompt(sub, merged, state), sub, state), merged)
            if not is_no_evidence_answer(retry):
                return retry + self._source_notice(self._has_local(merged), self._has_web(merged))
        if state.get('enable_web', True) and not has_web:
            web_docs = self._web_docs(sub, state)
            if web_docs:
                retry = self._generate(self._build_prompt(sub, web_docs, state), sub, state)
                retry = self._sanitize_images(retry, web_docs)
                if not is_no_evidence_answer(retry):
                    return retry + WEB_SOURCE_NOTICE
        return REFUSAL_TEMPLATE

    def _merge_docs(self, docs: List[Dict[str, Any]],
                    siblings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """兄弟子问证据并回本问上下文，按重排分降序，避免去重把关键切片留给另一个子问。"""
        merged, seen = [], set()
        for d in list(docs) + list(siblings):
            key = d.get('chunk_id') or id(d)
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
        merged.sort(key=lambda d: d.get('score') if d.get('score') is not None else (d.get('rrf_score') or 0.0),
                    reverse=True)
        return merged

    def _image_answer(self, sub: str, docs: List[Dict[str, Any]], state: QueryGraphState,
                      image_mode: str) -> str:
        """配图类子问：VL 读真实像素，图另起一段列出；一张都没有就如实说没有。"""
        refs = self._collect_refs(docs, fallback_all=True)
        if not refs:
            return IMAGE_NONE_TEMPLATE
        context, _ = self._format_retrieval_context(docs, self._sub_budget(len(state.get('sub_queries') or [1])))
        item_names = ",".join(state.get('item_names') or [])
        vl_text = self._vl_answer(sub, refs, context, item_names, state)
        lines = "\n\n".join(f"![{r['caption'] or '参考配图'}]({r['url']})" for r in refs)
        if image_mode == 'show':
            multi = len(state.get('sub_queries') or []) > 1
            heading = "" if multi else "# 相关配图\n\n"
            self.logger.info(f"配图直答：附 {len(refs)} 张相关图")
            return heading + (vl_text + "\n\n" if vl_text else "") + lines
        if vl_text:
            return vl_text + "\n\n" + lines
        # 读图失败：退回文档正文作答，仍保留配图
        return self._text_answer(sub, docs, state, has_web=False, has_local=True) + "\n\n" + lines

    def _vl_answer(self, sub: str, refs: List[Dict[str, Any]], context: str,
                   item_names: str, state: QueryGraphState) -> Optional[str]:
        task_id, is_stream = state.get('task_id'), state.get('is_stream')
        on_delta = (lambda d: push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": d})) \
            if is_stream else None
        return answer_with_images(sub, refs, context, item_names, on_delta=on_delta)

    # ==================== 上下文工程 ====================

    def _sub_budget(self, n_subs: int) -> int:
        """总预算按子问数均分，但不低于单问下限；单问直接用总预算。"""
        total = self.config.max_context_chars
        if n_subs <= 1:
            return total
        return max(self.config.sub_context_floor_chars, total // n_subs)

    def _dedup_docs(self, state: QueryGraphState, sub_queries: List[str]) -> List[List[Dict[str, Any]]]:
        """跨子问去重：同一 chunk 只服务第一个需要它的子问，全部重复时保留原样。"""
        per_docs = state.get('reranked_docs_per_q') or []
        out: List[List[Dict[str, Any]]] = []
        seen: set = set()
        for i in range(len(sub_queries)):
            docs = per_docs[i] if i < len(per_docs) else (state.get('reranked_docs') or [] if i == 0 else [])
            fresh = []
            for d in docs:
                key = d.get('chunk_id')
                if key and key in seen:
                    continue
                fresh.append(d)
            for d in fresh:
                if d.get('chunk_id'):
                    seen.add(d['chunk_id'])
            out.append(fresh if fresh else list(docs))
        return out

    def _build_prompt(self, user_query: str, docs: List[Dict[str, Any]], state: QueryGraphState) -> str:
        budget = self._sub_budget(len(state.get('sub_queries') or [1]))
        formatted_context, usage_chars = self._format_retrieval_context(docs, budget)
        formatted_history = self._format_chat_history(state.get('history') or [], usage_chars)
        return ANSWER_PROMPT.format(
            context=formatted_context or "暂无检索到上下文",
            history=formatted_history or "暂无历史上下文",
            item_names=",".join(state.get('item_names') or []),
            question=user_query,
            no_evidence_token="NO_EVIDENCE"
        )

    def _web_docs(self, sub: str, state: QueryGraphState) -> List[Dict[str, Any]]:
        """答案层兜底联网：把网页片段转成与重排节点同构的文档。"""
        pages = web_search_once(sub, state.get('item_names') or [])
        pairs = [(sub, p.get('snippet', '')) for p in pages if p.get('snippet')]
        scores = batch_score_pairs(pairs)
        out = []
        for page, score in zip([p for p in pages if p.get('snippet')], scores):
            out.append({"content": page['snippet'], "chunk_id": None, "title": page.get('title', ''),
                        "url": page.get('url', ''), "source": "web", "rrf_score": None, "score": score})
        return out

    @staticmethod
    def _has_local(docs: List[Dict[str, Any]]) -> bool:
        return any(d.get('source') == 'local' for d in docs)

    @staticmethod
    def _has_web(docs: List[Dict[str, Any]]) -> bool:
        return any(d.get('source') == 'web' for d in docs)

    @staticmethod
    def _source_notice(has_local: bool, has_web: bool) -> str:
        if has_web and has_local:
            return WEB_SOURCE_NOTICE_MIXED
        if has_web:
            return WEB_SOURCE_NOTICE
        return ""

    @staticmethod
    def _section_prefix(sub: str, index: int, multi: bool) -> str:
        if not multi:
            return ""
        return f"\n\n---\n\n## {sub}\n\n" if index > 0 else f"## {sub}\n\n"

    # ==================== 配图引用 ====================

    def _collect_refs(self, docs: List[Dict[str, Any]], fallback_all: bool) -> List[Dict[str, Any]]:
        """收集配图引用（去重、截断）。fallback_all=True 时无 matched_images 的切片退回全部图。"""
        refs, seen = [], set()
        limit = self.config.image_max_show
        for doc in docs:
            if doc.get('matched_images'):
                candidates = doc['matched_images']
            elif fallback_all:
                candidates = parse_image_refs(doc.get('content', ''))
            else:
                candidates = []
            for ref in candidates:
                if ref['url'] not in seen:
                    seen.add(ref['url'])
                    refs.append(ref)
            if len(refs) >= limit:
                break
        return refs[:limit]

    def _sanitize_images(self, text: str, docs: List[Dict[str, Any]]) -> str:
        """配图白名单 + 去裸 URL：只保留命中切片里的图，其余图片/裸链接一律剔除。"""
        allowed = set()
        for doc in docs[:self._IMAGE_DOC_TOP_N]:
            allowed.update(doc_image_urls(doc))
            for ref in doc.get('matched_images') or []:
                allowed.add(ref['url'])

        def md_repl(m):
            return m.group(0) if norm_img_url(m.group(2)) in allowed else ''
        text = self._IMG_MD_RE.sub(md_repl, text)

        if "【图片】" in text:
            head, _, tail = text.partition("【图片】")
            kept = []
            for line in tail.splitlines():
                u = norm_img_url(line)
                if u and u in allowed:
                    kept.append(u)
            text = head.rstrip()
            if kept:
                text += "\n\n" + "\n\n".join(f"![参考配图]({u})" for u in kept)

        text = re.sub(r"<(https?://[^>\s]+)>",
                      lambda m: m.group(0) if norm_img_url(m.group(1)) in allowed else '', text)

        def bare_repl(m):
            u = norm_img_url(m.group(0))
            if u in allowed and self._IMG_EXT_RE.search(u):
                return f"![参考配图]({u})"
            return ''
        text = re.sub(r"(?<!\()https?://[^\s<>）)\]]+", bare_repl, text)

        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = self._FENCE_RE.sub(self._drop_empty_heading_fence, text)
        return text.strip()

    def _drop_empty_heading_fence(self, m) -> str:
        lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
        if lines and all(l in self._EMPTY_HEADINGS for l in lines):
            return ''
        return m.group(0)

    # ==================== 上下文格式化 ====================

    def _format_retrieval_context(self, retrieval_context: List[Dict[str, Any]], max_context_chars: int) -> Tuple[str, int]:
        formatted_lines = []
        usage = 0
        for index, context in enumerate(retrieval_context, 1):
            content = context.get('content', "")
            if not content:
                continue
            metadata_content = [f"[文档:{index}]"]
            for meta_field, template in [("chunk_id", "[chunk_id={}]"),
                                         ("title", "[title={}]"),
                                         ("source", "[source={}]"),
                                         ("url", "[url={}]")]:
                filed_value = str(context.get(meta_field, "")).strip()
                if filed_value:
                    metadata_content.append(template.format(filed_value))
            doc_score = context.get('score')
            if doc_score is not None:
                metadata_content.append(f"[score={float(doc_score):.6f}]")
            formatted_line = " ".join(metadata_content) + "\n" + content
            sep_chars = 2 if formatted_lines else 0
            total_length = sep_chars + len(formatted_line)
            if usage + total_length > max_context_chars:
                break
            formatted_lines.append(formatted_line)
            usage += total_length
        return "\n\n".join(formatted_lines), max_context_chars - usage

    # ==================== LLM 调用 ====================

    def _generate(self, prompt: str, sub: str, state: QueryGraphState) -> str:
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"获取LLM客户端失败 原因:{str(e)}")
            return REFUSAL_TEMPLATE

        state['prompt'] = prompt
        if state.get('is_stream'):
            return self._stream_llm(state.get('task_id'), prompt, llm_client)
        return self._invoke_llm(prompt, llm_client)

    def _invoke_llm(self, prompt: str, llm_client: ChatOpenAI) -> str:
        try:
            llm_res = llm_client.invoke(prompt)
            return (getattr(llm_res, 'content', "") or "").strip()
        except Exception as e:
            self.logger.error(f"非流式调用LLM失败 原因:{str(e)}", exc_info=True)
            return "LLM暂无法回答"

    def _stream_llm(self, task_id, prompt, llm_client) -> str:
        """流式生成，但把可能是 NO_EVIDENCE 的开头扣住不发。

        无证据契约的输出只有一行、不超过 11 个字符，扣住它最多延迟一个 token，
        却避免了"已经推给前端一行 NO_EVIDENCE、后面又换成模板"的自相矛盾画面。
        """
        buffer, flushed = "", False
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, 'content', "") or ""
                if not delta_text:
                    continue
                buffer += delta_text
                if not flushed:
                    probe = buffer.strip().strip("`").strip().upper()
                    if any(p.startswith(probe) for p in self._NO_EVIDENCE_PREFIXES):
                        continue
                    flushed = True
                    push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": buffer})
                else:
                    push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式调用LLM失败 原因:{str(e)}", exc_info=True)
            return "LLM暂无法回答"
        if buffer and not flushed:
            push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": buffer})
        return buffer.strip()

    def save_history(self, state: QueryGraphState):
        session_id = state.get('session_id')
        user_query = state.get('original_query')
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names') or []
        try:
            save_chat_message(session_id=session_id, role="user", text=user_query,
                              rewritten_query=rewritten_query, item_names=item_names)
            save_chat_message(session_id=session_id, role="assistant", text=state.get('answer'),
                              rewritten_query=rewritten_query, item_names=item_names)
        except Exception as e:
            self.logger.error(f"保存历史对话到MongDB中失败 原因:{str(e)}")

    def _format_chat_history(self, chat_history_context: List[Dict[str, Any]], usage_chars: int) -> str:
        formatted_lines = []
        used_chars = 0
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in chat_history_context:
            role = msg.get('role', '')
            text = msg.get('text', '')
            if not text or role not in role_map:
                continue
            formatted_line = f"{role_map[role]}: {text}"
            seperator_usage = 1 if formatted_lines else 0
            total_usage = seperator_usage + len(formatted_line)
            if used_chars + total_usage > usage_chars:
                break
            formatted_lines.append(formatted_line)
            used_chars += total_usage
        return "\n".join(formatted_lines)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    node = AnswerOutPutNode()
    mock_state = {
        "original_query": "RS-12 数字万用表如何测量直流电压？",
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
        "sub_queries": ["RS-12 数字万用表如何测量直流电压？"],
        "query_plan": [{"image_mode": "none", "needs_web": "auto"}],
        "answer_mode_per_q": ["local"],
        "reranked_docs_per_q": [[{"content": "将旋钮拨到 DCV 档，红表笔接正极，读数即为直流电压。",
                                  "chunk_id": "c1", "title": "直流电压测量", "source": "local", "score": 0.9}]],
        "is_stream": False,
        "enable_web": False,
        "session_id": "smoke-test",
    }
    print(node.process(mock_state)['answer'])
