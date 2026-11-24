import asyncio
import json
import logging
from json import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Union

from agents.mcp import MCPServerStreamableHttp

from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState

logger = logging.getLogger(__name__)


def build_web_query(query: str, item_names: List[str]) -> str:
    """联网搜索词拼上产品名。短查询（如"外观是怎么样的"）不带产品名时会搜回
    大量无关文章（设计理论/汽车等），其重排分远高于本地产品切片，把本地正文/配图
    挤出上下文。产品名取自知识库登记名，对任意产品通用。"""
    product = " ".join(n for n in item_names if n)
    if product and product not in query:
        return f"{product} {query}"
    return query


async def _call_mcp_search(text: str) -> List[Dict[str, Any]]:
    """调用百炼 MCP 的 bailian_web_search 工具。"""
    from muti_rag.processor.query_processor.config import get_config

    cfg = get_config()
    async with MCPServerStreamableHttp(
            name="联网搜索",
            params={
                "url": cfg.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {cfg.openai_api_key}"},
                "timeout": 30,
            },
            cache_tools_list=True,
            max_retry_attempts=2,
    ) as mcp_client:
        result = await mcp_client.call_tool(tool_name="bailian_web_search",
                                            arguments={"query": text, "count": 3})

    text_content = result.content[0] if result.content else None
    if not text_content or not getattr(text_content, "text", ""):
        return []
    try:
        obj: Dict[str, Any] = json.loads(text_content.text)
    except JSONDecodeError as e:
        return []

    pages = obj.get("pages", []) or []
    return [{"title": str(p.get("title", "")).strip(),
             "snippet": str(p.get("snippet", "")).strip(),
             "url": str(p.get("url", "")).strip()} for p in pages]


def web_search_once(query: str, item_names: List[str]) -> List[Dict[str, Any]]:
    """单次联网检索（供联网节点与证据门控复用）。失败返回空列表，不抛出。"""
    text = build_web_query(query, item_names)
    try:
        return asyncio.run(_call_mcp_search(text)) or []
    except Exception as e:
        logger.warning(f"联网搜索失败 [{text}]: {e}")
        return []


class WebMcpSearchNode(BaseNode):
    """联网检索节点（前置分支）。

    V5 起不再"每问都联网"：只有查询理解标注 needs_web=yes 的子问题在这里同步检索；
    needs_web=auto 的子问题留给证据门控，等本地证据确认不足后再补联网，省掉一次
    无用的外部调用。图片类子问题始终跳过（联网综述回答不了"文档里有没有这张图"）。
    """

    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        if not state.get("enable_web", True):
            self.logger.info("用户关闭联网搜索，跳过联网分支")
            return {}

        item_names = state.get("item_names") or []
        subs = state.get("sub_queries") or [state.get("rewritten_query") or state.get("original_query")]
        plan = state.get("query_plan") or []

        todo = []
        for i, sub in enumerate(subs):
            meta = plan[i] if i < len(plan) else {}
            if meta.get("needs_web") != "yes":
                continue
            if meta.get("image_mode", "none") != "none":
                continue
            todo.append(i)

        per_q: List[List[Dict[str, Any]]] = [[] for _ in subs]
        if not todo:
            self.logger.info("无子问题需要同步联网检索（auto 交由证据门控按需补检索）")
            return {"web_search_docs_per_q": per_q, "web_search_docs": per_q[0] if per_q else []}

        with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
            futures = {i: pool.submit(web_search_once, subs[i], item_names) for i in todo}
            for i, fut in futures.items():
                per_q[i] = fut.result() or []

        self.logger.info(f"同步联网检索完成：{len(todo)} 个子问题，共 {sum(len(r) for r in per_q)} 条结果")
        return {"web_search_docs_per_q": per_q, "web_search_docs": per_q[0] if per_q else []}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(web_search_once("RS-12 数字万用表如何测量直流电压？", ["RS-12 数字万用表"]))
