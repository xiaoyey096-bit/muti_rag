"""证据判定工具。

背景（2026-11-18 标定，见 temp_data/eval_20261118/evidence_signals.py）：
- 重排绝对分不可用作阈值：库外题 top1 中位 0.96，可答题反而只有 0.013；
- 原始稠密余弦同样不可分：库外题 top1 落在 0.56~0.76，与可答题 0.55~0.79 完全重叠。
因此这里只用"确定性、可解释"的证据信号：
1 是否有召回（过滤条件命中为空 = 真的检索不到）；
2 词汇覆盖率：问题去掉实体词后的字符 bigram 有多少出现在候选切片里——
  覆盖率低说明问题里的关键名词根本没在文档中出现，是 HyDE 应该补召回的信号，
  也是"证据与问题无关"的信号。
"""

import re
import unicodedata
from typing import Iterable, List, Sequence

# 疑问/口语骨架词：这些词在任何文档里都可能出现，不能作为"证据相关"的依据
_STOP_TERMS = {"是什么", "怎么", "怎样", "如何", "请问", "多少", "哪个", "哪些", "有没有", "是否",
               "可以", "需要", "这个", "那个", "它", "他", "我们", "你们", "的话", "一下", "麻烦"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).lower()


def bigrams(text: str, strip_terms: Sequence[str] = ()) -> set:
    s = normalize(text)
    for term in strip_terms:
        term = normalize(term)
        if term:
            s = s.replace(term, "")
    for term in _STOP_TERMS:
        s = s.replace(term, "")
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)
    if len(s) <= 1:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def lexical_coverage(query: str, docs: Iterable[str], strip_terms: Sequence[str] = ()) -> float:
    """问题 bigram 在单个文档中的最大命中率。无有效 bigram 时返回 1（视为不构成弱证据信号）。"""
    grams = bigrams(query, strip_terms)
    if not grams:
        return 1.0
    best = 0.0
    for doc in docs:
        if not doc:
            continue
        hit = len(grams & bigrams(doc)) / len(grams)
        if hit > best:
            best = hit
    return best


def is_no_evidence_answer(text: str) -> bool:
    """模型按契约输出的"无证据"标记（容忍反引号/空白/句号）。"""
    body = re.sub(r"[`\"'.。\s]", "", normalize(text))
    return body == "no_evidence"
