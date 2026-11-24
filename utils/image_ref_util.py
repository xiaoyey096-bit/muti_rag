"""切片正文中的图片引用解析（markdown 与裸 URL 两种形态）。

导入期 md_to_img 会把图片写成 ![摘要](http://…)，个别历史切片是裸 http URL。
检索/重排/答案校验共用本模块，保证"含图判定、配图白名单、图题相关性打分"口径一致。
"""

import re
from typing import Any, Dict, List

_IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)]+)\)")
_IMG_PLAIN_RE = re.compile(r"https?://[^\s\"'<>)]+\.(?:png|jpe?g|gif|bmp|webp)", re.I)


def norm_img_url(url: str) -> str:
    return url.strip().strip("<>").replace(" ", "%20")


def parse_image_refs(content: str) -> List[Dict[str, str]]:
    """返回 [{"url","caption"}]，按出现顺序去重；裸 URL 的 caption 为空串。"""
    refs: List[Dict[str, str]] = []
    seen = set()
    text = str(content or "")
    for m in _IMG_MD_RE.finditer(text):
        url = norm_img_url(m.group(2))
        if url and url not in seen:
            seen.add(url)
            refs.append({"url": url, "caption": m.group(1).strip()})
    for m in _IMG_PLAIN_RE.finditer(text):
        url = norm_img_url(m.group(0))
        if url and url not in seen:
            seen.add(url)
            refs.append({"url": url, "caption": ""})
    return refs


def doc_image_urls(doc: Dict[str, Any]) -> List[str]:
    return [r["url"] for r in parse_image_refs(doc.get("content", ""))]