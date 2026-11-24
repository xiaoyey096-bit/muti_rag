"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.processor.query_processor.nodes.item_name_confirmed_node import ItemNameConfirmedNode
from muti_rag.processor.query_processor.nodes.local_search_node import LocalSearchNode
from muti_rag.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode
from muti_rag.processor.query_processor.nodes.rrf_merge_node import RrfMergeNode
from muti_rag.processor.query_processor.nodes.reranker_node import RerankerNode
from muti_rag.processor.query_processor.nodes.evidence_gate_node import EvidenceGateNode
from muti_rag.processor.query_processor.nodes.answer_output_node import AnswerOutPutNode

# 从 knowledge 目录加载环境变量
_env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(_env_path)


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    流程结构（V5）::

        item_name_confirmed_node          ← 一次 LLM 调用：实体/改写/意图/子问题拆分/策略标注
              │
              ├── (澄清短路) ─────────────────────────────> answer_output_node
              │                                                │
              └── (继续检索)                                    │
                   │                                           │
                   v                                           │
              multi_search  (虚拟分发节点)                       │
                   │                                           │
             ┌─────┴──────────┐                                │
             v                v                                │
      local_search_node  web_mcp_search_node                   │
      (混合召回→按需HyDE)   (仅 needs_web=yes)                   │
             │                │                                │
             └───────┬────────┘                                │
                     v                                         │
                   join                                        │
                     v                                         │
                rrf_merge_node      (逐子问融合)                 │
                     v                                         │
                reranker_node       (一次批量重排 + 召回顺序保底)  │
                     v                                         │
              evidence_gate_node    (逐子问：本地/联网/固定模板)   │
                     v                                         │
             answer_output_node <──────────────────────────────┘
                     v
                    END

    注：本地召回（混合 + HyDE）合并在 local_search_node 内串行，是为了让图上与联网分支
    同深度——汇合节点是"任一上游完成即触发"，深度不等的分支会让 join 及其下游重复执行。
    """

    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)  # type:ignore

    # 2. 实例化节点
    nodes = {
        "item_name_confirmed_node": ItemNameConfirmedNode(),
        "multi_search": lambda x: x,  # 虚拟节点
        "local_search_node": LocalSearchNode(),
        "web_mcp_search_node": WebMcpSearchNode(),
        "join": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf_merge_node": RrfMergeNode(),
        "reranker_node": RerankerNode(),
        "evidence_gate_node": EvidenceGateNode(),
        "answer_output_node": AnswerOutPutNode(),
    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)  # type:ignore

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirmed_node")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirmed_node",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output_node",
        },
    )

    # 6. 多路搜索分发（本地召回与联网检索同深度并行；子问题拆分已在查询理解里完成）
    workflow.add_edge("multi_search", "local_search_node")
    workflow.add_edge("multi_search", "web_mcp_search_node")

    # 7. 多路搜索汇合
    workflow.add_edge("local_search_node", "join")
    workflow.add_edge("web_mcp_search_node", "join")

    # 8. 顺序边
    workflow.add_edge("join", "rrf_merge_node")
    workflow.add_edge("rrf_merge_node", "reranker_node")
    workflow.add_edge("reranker_node", "evidence_gate_node")
    workflow.add_edge("evidence_gate_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    # 9. 返回可运行的状态
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()

if __name__ == "__main__":
    import time

    from pymilvus import MilvusClient

    from muti_rag.processor.query_processor.config import get_config


    def known_item_name() -> str:
        """从 kb_item_names_v1 读出当前真实入库的商品名。

        商品名是导入时由 LLM 从文档里抽取的，同一篇 PDF 不同批次可能抽成
        "数字万用表" 或 "RS-12 数字万用表"。写死在测试里会让两个场景的预期
        随着导入结果翻转，所以这里改成运行时实时读取。
        """
        cfg = get_config()
        try:
            client = MilvusClient(uri=cfg.milvus_url)
            if not client.has_collection(cfg.item_name_collection):
                return ""
            client.load_collection(cfg.item_name_collection)
            rows = client.query(collection_name=cfg.item_name_collection, filter="",
                                output_fields=["item_name"], limit=10)
            return str(rows[0]["item_name"]) if rows else ""
        except Exception as exc:
            print(f"  [warn] 读取 {cfg.item_name_collection} 失败，退回用文件标题: {exc}")
            return ""


    ITEM_NAME = known_item_name() or "万用表的使用"

    print("=" * 60)
    print("开始测试: 查询流程主图 (main_graph)")
    print(f"向量库中当前商品名: {ITEM_NAME}")
    print("=" * 60)

    # 每次运行用唯一 session_id：历史对话存在 MongoDB 里，
    # 固定 session 会跨次运行累积，反过来影响商品名提取与查询重写，导致测试不可复现
    run_tag = str(int(time.time()))

    # ---- 测试场景 1：商品名与向量库一致，走完整 pipeline ----
    # 注意：query 里的商品名必须与 kb_item_names_v1 中的记录一致，否则会被
    #       item_name_confirmed_node 拦截而短路，三路检索/RRF/Rerank 都不会执行
    print("\n【场景 1】: 商品名与向量库一致，走完整 pipeline")
    print("-" * 60)

    mock_state_1 = {
        "original_query": f"{ITEM_NAME}怎么测量直流电压？",
        "session_id": f"test_session_main_graph_{run_tag}",
        "task_id": f"test_task_001_{run_tag}",
        "is_stream": False,
    }

    print(f"  查询: {mock_state_1['original_query']}")

    result_1 = query_app.invoke(mock_state_1)

    print(f"\n  【结果】:")
    print(f"  商品名: {result_1.get('item_names')}")
    print(f"  重写查询: {result_1.get('rewritten_query')}")
    print(f"  本地召回(RRF后): {len(result_1.get('rrf_chunks') or [])} 条")
    print(f"  网络召回: {len(result_1.get('web_search_docs') or [])} 条")
    print(f"  Rerank后送入LLM: {len(result_1.get('reranked_docs') or [])} 条")
    for index, doc in enumerate(result_1.get('reranked_docs') or [], 1):
        score = doc.get('score')
        rrf_score = doc.get('rrf_score')
        print(f"    {index}. rerank={score if score is None else round(score, 4)} "
              f"rrf={rrf_score if rrf_score is None else round(rrf_score, 6)} "
              f"source={doc.get('source')} title={str(doc.get('title'))[:30]}")
    answer_1 = result_1.get("answer", "") or ""
    print(f"  答案: {answer_1[:200]}..." if len(answer_1) > 200 else f"  答案: {answer_1}")

    # ---- 测试场景 2：商品名无法与向量库对齐，被拦截直接返回候选 ----
    # 故意问一个库里没有的型号，稳定触发"澄清"分支（不走三路检索）
    print("\n\n【场景 2】: 商品名与向量库不一致，触发澄清")
    print("-" * 60)

    mock_state_2 = {
        "original_query": "福禄克 Fluke 17B 万用表如何测量直流电压？",
        "session_id": f"test_session_main_graph_{run_tag}_2",
        "task_id": f"test_task_002_{run_tag}",
        "is_stream": False,
    }

    print(f"  查询: {mock_state_2['original_query']}")

    result_2 = query_app.invoke(mock_state_2)

    print(f"\n  【结果】:")
    print(f"  商品名: {result_2.get('item_names')}")
    print(f"  重写查询: {result_2.get('rewritten_query')}")
    print(f"  是否进入三路检索: {bool(result_2.get('rrf_chunks'))}")
    answer_2 = result_2.get("answer", "") or ""
    print(f"  答案: {answer_2}")

    print("\n" + "=" * 60)
    print("全部测试完成")
