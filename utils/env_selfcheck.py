import logging

logger = logging.getLogger(__name__)


def ensure_pypdf_stream_limit(min_bytes: int = 500_000_000) -> None:
    """
    启动自检：pypdf 默认允许的 PDF 流长度是 75MB，大扫描件会直接 LimitReachedError。
    Pip 升级 pypdf 会还原默认值，这里在进程启动时兜底提升并记录日志。
    注：MinerU 子进程由 sitecustomize.py 承担同样的兜底。
    """
    try:
        import pypdf.filters as filters
        import pypdf._configuration as cfg
    except Exception as e:
        logger.warning(f"pypdf 自检跳过：{e}")
        return

    changed = []
    try:
        if filters.MAX_DECLARED_STREAM_LENGTH < min_bytes:
            filters.MAX_DECLARED_STREAM_LENGTH = min_bytes
            changed.append("filters.MAX_DECLARED_STREAM_LENGTH")
        for attr in ("DEFAULT_CONFIGURATION", "CURRENT_CONFIGURATION"):
            conf = getattr(cfg, attr, None)
            if conf is not None and getattr(conf, "maximum_declared_stream_length", 0) < min_bytes:
                try:
                    cfg.overwrite_configuration(maximum_declared_stream_length=min_bytes)
                except Exception:
                    object.__setattr__(conf, "maximum_declared_stream_length", min_bytes)
                changed.append(attr)
    except Exception as e:
        logger.warning(f"pypdf 流长度上限自检失败：{e}")
        return

    if changed:
        logger.warning(f"已提升 pypdf 流长度上限到 {min_bytes}（{', '.join(changed)}），"
                       f"建议锁定 pypdf 版本避免升级后还原")
    else:
        logger.info(f"pypdf 流长度上限正常（{filters.MAX_DECLARED_STREAM_LENGTH}）")


def ensure_transformers_safe_load() -> None:
    """
    启动自检：transformers 4.5x 恒定使用 meta-device + mmap 权重加载，
    本机实测（torch 2.11 / Windows）在多线程算子下读取 mmap 权重会触发
    access violation 导致整进程崩溃，而单线程正常。
    这里只关闭并行分片加载；CPU 推理的单线程限制在使用方（ai_clients）设置。
    """
    import os
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info("transformers 加载自检完成（并行分片加载已关闭）")
