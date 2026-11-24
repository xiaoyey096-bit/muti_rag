"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """
    Represents the state of our query graph.
    Attributes:
    各个属性的结构
    """
    session_id: str # 会话ID
    task_id:str # 任务ID
    message_id: str # 消息ID
    original_query: str # 原始查询
    embedding_chunks: list # 已向量化的切片
    hyde_embedding_chunks: list # 已向量化的假设性问题切片
    rrf_chunks: list # rrf排序后的切片
    web_search_docs: list # 搜索结果
    reranked_docs: list  # 排序后的文档
    prompt: str  #提示词
    answer: str #答案
    item_names: List[str] # 商品名称
    kb_id: str # 所属知识库ID（多产品隔离）
    rewritten_query: str  #重写答案
    history: list   # 历史对话
    is_stream: bool # 是否流式输出
    enable_web: bool # 是否启用联网搜索分支
    sub_queries: List[str] # 拆分后的独立子问题（单问时为单元素列表）
    intent: str # 查询理解节点判定的主意图
    query_plan: list # 每个子问题的执行策略：query/intent/image_mode/use_hyde/needs_web
    evidence_per_q: list # 每个子问题的证据统计（本地/联网召回数、保留数、最强信号）
    answer_mode_per_q: list # 每个子问题的回答模式：local / web / refusal
    answer_mode: str # 整问的回答模式（多子问模式不同时为 mixed），用于观测与评测统计
    embedding_chunks_per_q: list # 每个子问题独立的混合检索结果
    hyde_embedding_chunks_per_q: list # 每个子问题独立的 HyDE 检索结果
    web_search_docs_per_q: list # 每个子问题独立的联网检索结果
    rrf_chunks_per_q: list # 每个子问题独立的 RRF 融合结果
    reranked_docs_per_q: list # 每个子问题独立的重排结果（各自 top3）


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",               # 会话ID
    "task_id": "",               # 任务ID
    "message_id": "",               # 消息ID
    "original_query": "",           # 原始查询
    "embedding_chunks": [],         # 已向量化的切片
    "hyde_embedding_chunks": [],    # 已向量化的假设性问题切片
    "rrf_chunks": [],               # rrf排序后的切片
    "web_search_docs": [],          # 搜索结果
    "reranked_docs": [],            # 排序后的文档
    "prompt": "",                   # 提示词
    "answer": "",                   # 答案
    "item_names": [],               # 商品名称
    "kb_id": "",                    # 所属知识库ID
    "rewritten_query": "",          # 重写查询
    "history": [],                  # 历史对话
    "is_stream": False,             # 是否流式输出 (默认设为 False)
    "enable_web": True,             # 是否启用联网搜索分支 (默认开启)
    "sub_queries": [],              # 拆分后的独立子问题
    "intent": "",                   # 主意图
    "query_plan": [],               # 子问题执行策略
    "evidence_per_q": [],           # 每问证据统计
    "answer_mode_per_q": [],        # 每问回答模式 local/web/refusal
    "answer_mode": "",              # 整问回答模式（子问模式不一致时为 mixed）
    "embedding_chunks_per_q": [],   # 每问独立混合检索
    "hyde_embedding_chunks_per_q": [],  # 每问独立 HyDE
    "web_search_docs_per_q": [],    # 每问独立联网
    "rrf_chunks_per_q": [],         # 每问独立 RRF
    "reranked_docs_per_q": [],      # 每问独立重排 top3
}

def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。

    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本。

    Returns:
        状态副本，避免修改全局默认值。
    """
    return copy.deepcopy(DEFAULT_STATE)


graph_default_state = DEFAULT_STATE
