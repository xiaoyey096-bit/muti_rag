import json
import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from muti_rag.processor.import_processor.base import BaseNode
from muti_rag.processor.import_processor.state import ImportGraphState
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils import kb_registry_util as kb_registry
from muti_rag.utils.image_ref_util import parse_image_refs
from muti_rag.prompts.import_prompt import (
    SUGGEST_QUESTIONS_SYSTEM_PROMPT,
    SUGGEST_QUESTIONS_USER_PROMPT_TEMPLATE,
)

_CONTEXT_LIMIT = 3500
_MAX_QUESTIONS = 3


class SuggestQuestionsNode(BaseNode):
    """导入完成后让 LLM 预测用户常问的 3 个问题，写入知识库注册表，
    供问答入口页渲染成可点击的快速提问按钮。失败只记日志，不影响导入主流程。"""

    name = "suggest_questions_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        kb_id = state.get('kb_id', '')
        if not kb_id:
            self.logger.info("无 kb_id，跳过推荐问题生成")
            return state

        try:
            questions = self._generate(state)
        except Exception as e:
            self.logger.error(f"推荐问题生成失败（不影响导入）: {e}")
            return state

        if not questions:
            self.logger.warning("推荐问题生成结果为空，跳过写入")
            return state

        try:
            kb_registry.set_suggested_questions(kb_id, questions)
        except Exception as e:
            self.logger.error(f"推荐问题写入注册表失败 kb_id={kb_id}: {e}")
            return state

        state['suggested_questions'] = questions
        self.logger.info(f"kb_id={kb_id} 推荐问题: {questions}")
        return state

    def _generate(self, state: ImportGraphState) -> List[str]:
        item_name = state.get('item_name') or state.get('file_title') or ''
        file_title = state.get('file_title') or ''
        context = self._build_context(state)
        if not context.strip():
            return []

        llm = AIClients.get_llm_client(response_format=True)
        user_prompt = SUGGEST_QUESTIONS_USER_PROMPT_TEMPLATE.format(
            item_name=item_name, file_title=file_title, context=context,
            image_captions=self._collect_captions(state))
        resp = llm.invoke([
            SystemMessage(content=SUGGEST_QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return self._parse_questions(resp.content)

    def _collect_captions(self, state: ImportGraphState) -> str:
        captions, seen = [], set()
        for chunk in state.get('chunks') or []:
            content = chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
            for ref in parse_image_refs(content):
                key = ref['caption'] or ref['url']
                if key in seen:
                    continue
                seen.add(key)
                captions.append(ref['caption'] or ref['url'].rsplit('/', 1)[-1])
        if not captions:
            return "无（该文档不含任何配图，不要生成配图/外观类问题）"
        return "\n".join(f"- {c}" for c in captions[:12])

    def _build_context(self, state: ImportGraphState) -> str:
        # 均匀多点采样：首/尾常是目录、安全须知、页脚等样板，多点覆盖才能命中正文
        chunks = [c for c in (state.get('chunks') or []) if isinstance(c, dict) and c.get('content')]
        if chunks:
            n = len(chunks)
            idxs = sorted({min(n - 1, round(i * (n - 1) / 5)) for i in range(6)})
            return "\n".join(str(chunks[i].get('content'))[:600] for i in idxs)[:_CONTEXT_LIMIT]
        md_content = state.get('md_content') or ''
        return md_content[:_CONTEXT_LIMIT]

    @staticmethod
    def _parse_questions(text: str) -> List[str]:
        text = (text or '').strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get('questions') or []
        if not isinstance(data, list):
            return []
        questions, seen = [], set()
        for q in data:
            if not isinstance(q, str):
                continue
            q = q.strip()
            if not q or q in seen:
                continue
            seen.add(q)
            questions.append(q)
            if len(questions) >= _MAX_QUESTIONS:
                break
        return questions