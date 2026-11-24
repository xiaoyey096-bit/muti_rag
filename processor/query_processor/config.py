"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path

from dotenv import load_dotenv
# 从 knowledge 目录加载 .env
_env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(_env_path)


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )
    # 多问时单问上下文预算下限：总预算按子问数均分，但不低于该值
    sub_context_floor_chars: int = field(
        default_factory=lambda: int(os.getenv("SUB_CONTEXT_FLOOR_CHARS", "3000"))
    )

    # ==================== 查询理解（意图路由）配置 ====================
    max_sub_queries: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUB_QUERIES", "4"))
    )
    # 子问近义判重的相似度阈值（difflib 序列比）
    sub_query_dedup_ratio: float = field(
        default_factory=lambda: float(os.getenv("SUB_QUERY_DEDUP_RATIO", "0.9"))
    )

    # ==================== HyDE 自适应配置 ====================
    hyde_enabled: bool = field(
        default_factory=lambda: os.getenv("HYDE_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    # 词汇覆盖率低于该值说明问题的关键词没在候选切片里出现，混合召回不足以定位答案，补 HyDE
    hyde_coverage_floor: float = field(
        default_factory=lambda: float(os.getenv("HYDE_COVERAGE_FLOOR", "0.35"))
    )
    hyde_doc_chars: int = field(
        default_factory=lambda: int(os.getenv("HYDE_DOC_CHARS", "120"))
    )
    hyde_cache_ttl_s: int = field(
        default_factory=lambda: int(os.getenv("HYDE_CACHE_TTL_S", "600"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )
    # 重排绝对分不可信（实测库外题 top1 中位 0.96、可答题 0.013），因此按召回顺序保底：
    # RRF 排名前 N 的本地切片一定进入上下文，避免断崖把正确答案切掉。
    # 取 4 是实测定的：RS-12 电池型号那篇切片排在召回第 3~4 位，keep=2 时被断崖切掉后
    # 直接判无证据拒答；keep=4 后答对，且库外题（保修期）在保留 6 篇噪声切片时仍正确拒答。
    local_recall_keep: int = field(
        default_factory=lambda: int(os.getenv("LOCAL_RECALL_KEEP", "4"))
    )

    # ==================== 配图门控配置 ====================
    image_max_show: int = field(
        default_factory=lambda: int(os.getenv("IMAGE_MAX_SHOW", "2"))
    )
    image_caption_floor: float = field(
        default_factory=lambda: float(os.getenv("IMAGE_CAPTION_FLOOR", "0.5"))
    )
    # 没有任何图题过强门槛时，保留图题分最高的 1 张作为低置信配图（避免"有图说成没图"）
    image_caption_weak_floor: float = field(
        default_factory=lambda: float(os.getenv("IMAGE_CAPTION_WEAK_FLOOR", "0.2"))
    )
    image_max_side: int = field(
        default_factory=lambda: int(os.getenv("IMAGE_MAX_SIDE", "1024"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "5"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.75"))
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.45"))
    )
    item_name_score_gap: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SCORE_GAP", "0.08"))
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "3"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config