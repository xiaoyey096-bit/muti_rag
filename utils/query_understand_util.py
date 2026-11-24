"""查询理解（V5 统一意图路由）。

一次 LLM 调用同时产出：实体、自包含改写、意图、语义子问题与每个子问题的执行策略
（是否 HyDE、是否需要联网、图片模式）。下游节点不再各自用正则猜意图，也不再单独
调用拆分模型——拆分只由语义决定，标点不作为拆分依据。

调用失败时退化为"单问题 + auto 策略"，并用正则兜底识别图片意图，保证链路不中断。
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from muti_rag.prompts.query_prompt import QUERY_UNDERSTAND_PROMPT
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.query_intent_util import is_image_query

logger = logging.getLogger(__name__)

_INTENTS = {"fact", "compare", "image", "chat", "out_of_scope"}
_FLAGS = {"yes", "no", "auto"}
_IMAGE_MODES = {"show", "read", "none"}
# 模型偶尔会把提示词里的占位示例当成识别结果返回，直接进检索过滤会召回为空
_PLACEHOLDERS = {"产品名", "产品名称", "识别到的具体产品型号或名称", "无", "未知", "none", "null", "n/a", "unknown"}


@dataclass
class SubPlan:
    """单个子问题的执行策略。"""

    query: str
    intent: str = "fact"
    use_hyde: str = "auto"
    needs_web: str = "auto"
    image_mode: str = "none"
    entities: List[str] = field(default_factory=list)

    @property
    def wants_image(self) -> bool:
        return self.image_mode in ("show", "read")


@dataclass
class QueryPlan:
    """整条查询的理解结果。"""

    item_names: List[str]
    rewritten_query: str
    intent: str
    subs: List[SubPlan]
    fallback: bool = False


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm_key(text: str) -> str:
    return re.sub(r"[\s，,。.、？?！!；;：:]+", "", _clean(text)).lower()


def _one_flag(value: Any, allowed: set, default: str) -> str:
    v = _clean(value).lower()
    return v if v in allowed else default


def _fallback_plan(query: str, item_names: Optional[List[str]] = None) -> QueryPlan:
    """LLM 不可用时的保守计划：不拆分，策略全部交给检索阶段判断。"""
    text = _clean(query)
    image_mode = "show" if is_image_query(text) else "none"
    names = [n for n in (item_names or []) if _clean(n)]
    return QueryPlan(
        item_names=names,
        rewritten_query=text,
        intent="image" if image_mode != "none" else "fact",
        subs=[SubPlan(query=text, intent="image" if image_mode != "none" else "fact",
                      use_hyde="auto", needs_web="auto", image_mode=image_mode, entities=names)],
        fallback=True,
    )


def _parse(raw: str, query: str) -> Optional[Dict[str, Any]]:
    content = re.sub(r"^```(?:json)?\s*", "", _clean(raw))
    content = re.sub(r"\s*```$", "", content)
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"查询理解输出无法解析: {e} / {content[:200]}")
        return None
    return obj if isinstance(obj, dict) else None


def _dedup_and_merge(subs: List[SubPlan], ratio: float, limit: int) -> List[SubPlan]:
    """去重 + 近义合并（同一信息需求被拆成两问时只留第一问）。"""
    kept: List[SubPlan] = []
    keys: List[str] = []
    for sub in subs:
        key = _norm_key(sub.query)
        if not key:
            continue
        merged = False
        for i, exist in enumerate(keys):
            if key == exist or difflib.SequenceMatcher(None, key, exist).ratio() >= ratio:
                logger.info(f"子问去重：丢弃「{sub.query}」，与「{kept[i].query}」近义")
                merged = True
                break
        if not merged:
            kept.append(sub)
            keys.append(key)
    return kept[:limit]


def _backfill_entities(subs: List[SubPlan], item_names: List[str], fallback_query: str) -> List[SubPlan]:
    """子问丢失产品实体时回填：省略主语的子问单独检索会召回大量无关内容。"""
    if not item_names:
        return subs
    primary = item_names[0]
    pool = [_norm_key(n) for n in item_names if _norm_key(n)]
    for sub in subs:
        key = _norm_key(sub.query)
        if pool and not any(p in key for p in pool):
            sub.query = f"{primary} {sub.query}".strip()
            logger.info(f"子问实体回填: {sub.query}")
        if not sub.query:
            sub.query = fallback_query
    return subs


def _clean_item_names(values: Any) -> List[str]:
    names = [_clean(n) for n in (values or []) if isinstance(n, str) and _clean(n)]
    kept = [n for n in names if n.lower() not in _PLACEHOLDERS and n not in _PLACEHOLDERS]
    if len(kept) != len(names):
        logger.info(f"丢弃占位商品名: {[n for n in names if n not in kept]}")
    return kept


def _normalize(obj: Dict[str, Any], original_query: str, max_subs: int, dedup_ratio: float) -> Optional[QueryPlan]:
    item_names = _clean_item_names(obj.get("item_names"))
    rewritten = _clean(obj.get("rewritten_query")) or _clean(original_query)
    intent = _one_flag(obj.get("intent"), _INTENTS, "fact")

    raw_subs = obj.get("sub_questions")
    subs: List[SubPlan] = []
    if isinstance(raw_subs, list):
        for item in raw_subs:
            if isinstance(item, str):
                item = {"query": item}
            if not isinstance(item, dict):
                continue
            text = _clean(item.get("query"))
            if not text:
                continue
            subs.append(SubPlan(
                query=text,
                intent=_one_flag(item.get("intent"), _INTENTS, intent),
                use_hyde=_one_flag(item.get("use_hyde"), _FLAGS, "auto"),
                needs_web=_one_flag(item.get("needs_web"), _FLAGS, "auto"),
                image_mode=_one_flag(item.get("image_mode"), _IMAGE_MODES,
                                    "show" if item.get("intent") == "image" else "none"),
                entities=item_names,
            ))

    if not subs:
        subs = [SubPlan(query=rewritten, intent=intent, use_hyde="auto", needs_web="auto",
                        image_mode="show" if intent == "image" else "none", entities=item_names)]

    subs = _dedup_and_merge(subs, dedup_ratio, max_subs)
    subs = _backfill_entities(subs, item_names, rewritten)
    return QueryPlan(item_names=item_names, rewritten_query=subs[0].query if len(subs) == 1 else rewritten,
                     intent=intent, subs=subs)


def understand_query(query: str, history_text: str = "", known_items: Optional[List[str]] = None,
                     max_subs: int = 4, dedup_ratio: float = 0.9) -> QueryPlan:
    """执行一次查询理解调用并输出可执行的策略计划。"""
    items = [n for n in (known_items or []) if _clean(n)]
    system = "你是产品知识库的查询理解模块，只输出 JSON。"
    user = QUERY_UNDERSTAND_PROMPT.format(
        known_items="、".join(items) if items else "未指定",
        max_subs=max_subs,
        history_text=_clean(history_text) or "暂无历史上下文",
        query=_clean(query),
    )

    try:
        client = AIClients.get_llm_client(response_format=True)
        resp = client.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as e:
        logger.warning(f"查询理解 LLM 调用失败，退化为单问题计划: {e}")
        return _fallback_plan(query, items)

    parsed = _parse(getattr(resp, "content", "") or "", query)
    if not parsed:
        return _fallback_plan(query, items)

    plan = _normalize(parsed, query, max_subs, dedup_ratio)
    if not plan:
        return _fallback_plan(query, items)

    # 已选定产品时以知识库登记名为准，避免模型改写完实体后与库内商品名不一致
    if items:
        plan.item_names = items
        for sub in plan.subs:
            sub.entities = items

    logger.info(f"查询理解: intent={plan.intent} 子问数={len(plan.subs)} "
                f"策略={[(s.use_hyde, s.needs_web, s.image_mode) for s in plan.subs]} "
                f"fallback={plan.fallback}")
    return plan
