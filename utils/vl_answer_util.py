"""查询期识图：把命中配图的真实像素 + 问题送 VL 模型作答。

MinIO 图片地址是内网 URL，DashScope 服务端无法回源抓取，
必须由本机下载图片字节后转 base64 data URL 再传给 VL 模型。
失败返回 None，由调用方回退原文本逻辑，不阻断问答。
"""
import os
import base64
import io
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import httpx

from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.client.base import logger
from muti_rag.prompts.query_prompt import VL_ANSWER_SYSTEM_PROMPT, VL_ANSWER_USER_TEMPLATE

_FETCH_TIMEOUT = 15
_MIME_BY_EXT = {'.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp'}


def _image_limit() -> int:
    from muti_rag.processor.query_processor.config import get_config
    return get_config().image_max_show


def _downscale(data: bytes) -> bytes:
    """等比缩到最长边 ≤ image_max_side：VL 模型按像素分块计费，文档截图 2K 缩到
    1K 后读数标识仍清晰，但请求体和首 token 延迟明显下降。"""
    from muti_rag.processor.query_processor.config import get_config
    max_side = get_config().image_max_side
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) <= max_side:
                return data
            ratio = max_side / max(im.size)
            resized = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))), Image.LANCZOS)
            buf = io.BytesIO()
            resized.convert('RGB').save(buf, format='JPEG', quality=88)
            return buf.getvalue()
    except Exception as e:
        logger.warning(f"配图缩放失败，按原图送 VL：{e}")
        return data


def _fetch_image_data_url(url: str) -> Optional[str]:
    try:
        resp = httpx.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        mime = _MIME_BY_EXT.get(ext, 'image/jpeg')
        data = _downscale(resp.content)
        return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"
    except Exception as e:
        logger.warning(f"识图下载图片失败 {url}: {e}")
        return None


def answer_with_images(question: str, refs: List[Dict[str, Any]], context: str,
                       item_names: str, on_delta: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """读图作答。refs=[{'url','caption'}]；成功返回 Markdown 文本，失败/无图返回 None。"""
    picked = refs[:_image_limit()]
    # 并行拉图：内网 MinIO 单张下载几百毫秒，串行会直接叠加到识图路径首字延迟
    if len(picked) == 1:
        data_urls = [_fetch_image_data_url(picked[0]['url'])]
    else:
        with ThreadPoolExecutor(max_workers=len(picked)) as pool:
            data_urls = list(pool.map(lambda r: _fetch_image_data_url(r['url']), picked))

    content: List[Dict[str, Any]] = []
    captions: List[str] = []
    for ref, data_url in zip(picked, data_urls):
        if not data_url:
            continue
        captions.append(ref.get('caption') or '参考配图')
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    if not content:
        return None

    text = VL_ANSWER_USER_TEMPLATE.format(
        item_names=item_names or '未指定',
        question=question,
        captions="\n".join(f"{i}. {c}" for i, c in enumerate(captions, 1)),
        context=context or '无',
    )
    messages = [
        {"role": "system", "content": VL_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": text}] + content},
    ]
    model = os.getenv("VL_MODEL", "qwen3-vl-plus")
    try:
        client = AIClients.get_vlm_client()
        if on_delta:
            pieces: List[str] = []
            stream = client.chat.completions.create(model=model, messages=messages, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    pieces.append(delta)
                    on_delta(delta)
            return "".join(pieces) or None
        resp = client.chat.completions.create(model=model, messages=messages)
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        logger.error(f"VL 识图回答失败: {e}")
        return None