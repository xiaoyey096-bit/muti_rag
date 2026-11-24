import logging, re, json
from dataclasses import asdict
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Dict, Tuple, List, Any
from muti_rag.processor.query_processor.base import BaseNode
from muti_rag.processor.query_processor.state import QueryGraphState
from muti_rag.utils.client.ai_clients import AIClients
from muti_rag.utils.client.storage_clients import StorageClients
from muti_rag.utils.query_understand_util import understand_query, QueryPlan
from muti_rag.utils.embedding_util import generate_bge_m3_hybrid_vectors
from muti_rag.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, build_search_filter
from muti_rag.processor.query_processor.base import get_config
from muti_rag.utils.mongo_history_util import get_recent_messages
from muti_rag.utils.kb_registry_util import get_kb


class _ItemNameAligner:

    def __init__(self):
        self._config = get_config()

    def search_and_align(self, item_names: List[str], kb_id: str = "") -> Tuple[List[str], List[str]]:
        """
        检索向量数据库并且和向量数据库中的商品名对齐 最终返回确定的商品名列表或者模糊的商品名列表
        Args:
            item_names: LLM提起到商品名列表
            kb_id: 知识库ID（提供时仅在该知识库范围内对齐）

        Returns:

        """

        # 1. 混合检索向量数据库
        search_result: List[Dict[str, Any]] = self._search_vector(item_names, kb_id)
        if not search_result:
            return [], []
        # 2. 根据混合向量检索到结果做对齐【confirmed/options 】
        confirmed, options = self._align(search_result)

        # 3. 分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        # 4. 返回确定的confirmed容器和options容器
        return confirmed, options

    def _search_vector(self, item_names: List[str], kb_id: str = "") -> List[Dict[str, Any]]:
        """
         对LLM提取到的所有商品名进行向量检索
        Args:
            item_names: LLM提取到商品名列表

        Returns:
          Dict[str, Any]:
          例子：{"extracted_name":"LLM提取的商品名1","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}
          例子：{"extracted_name":"LLM提取的商品名2","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}

        """
        final_search_result = []
        # 1. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return final_search_result
        # 2. 获取BGE-M3嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"BGE-M3客户端获取失败 原因:{str(e)}")
            return final_search_result

        # 3. 商品名列表向量化(混合向量)
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_m3_client, embedding_documents=item_names)
        except Exception as e:
            logger.error(f"商品列表{item_names}生成混合向量失败 原因:{str(e)} ")
            return final_search_result

        # 4. 混合向量检索（提供 kb_id 时仅在该知识库范围内对齐商品名）
        expr, expr_params = build_search_filter(kb_id=kb_id)
        for index, item_name in enumerate(item_names):
            # 4.1 构建稠密以及混合向量的检索请求
            hybrid_requests = create_hybrid_search_requests(hybrid_vector_result['dense'][index],
                                                            hybrid_vector_result['sparse'][index],
                                                            expr=expr,
                                                            expr_params=expr_params)

            # 4.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(milvus_client=milvus_client,
                                                               collection_name=self._config.item_name_collection,
                                                               search_requests=hybrid_requests,
                                                               ranker_weights=(0.5, 0.5),
                                                               norm_score=True,
                                                               limit=5,
                                                               output_fields=['item_name']
                                                               )

            # 4.3 解析结果

            matches = [{"score": item_search_res['distance'], "item_name": item_search_res['entity']['item_name']} for
                       item_search_res in
                       (hybrid_search_result[0] if hybrid_search_result else [])]

            # 4.4 封装
            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        主要职责：对齐
        怎么对齐？
        将什么样的商品名item_name放到confirmed
        将什么样的商品名item_name放到options
        将什么样的商品名item_name两个容器都不放
        制定规则：
        两个规则：1.如果从向量数据库中查到的商品名分数比如大于0.7 放到confirmed
        两个规则：2.如果从向量数据库中查到的商品名分数比如小于等于0.7大于0.45 放到options
                 3.如果从向量数据库中查到的商品名分数小于等于0.45 两个容器都不放
        疑问：TODO (后续RAG阈值调参)
         0.7 or 0.45如何给的?不应该拍着脑袋给。压测得到（构建数据集【询问的方式、llm提取到的商品名】 2. 构建阈值集 3.跑完整个流程：得到哪些阈值更适合构建的数据集）

        confirmed容器不应该出现两个一模一样的商品名--->下游检索【根本不需要两个一模一样的商品名】
        options容器中不能出现两个一模一样的商品名--->用户展示[也不应该展示两个一模一样的商品名]
        场景：confirmed中的商品名可能在options出现：如果某个商品名在confirmed中出现，不用在添加到options中。
        记住：
        像options中添加的商品名满足三个条件 条件1：小于high阈值且大于options 条件2：该商品名不在options 条件3：不在confirmed中
        像confirmed中添加商品名的:条件1：阈值大于high(3个小条件) 条件2：该商品名不在confirmed中 条件3：不用考虑options的，哪怕这个商品名已经在options.只要能进入不到
        confirmed中添加



        Args:
            search_result:  向量数据库检索到的结果

        Returns:
          最终两个容器confirmed、options列表中的商品名
        """

        # 1. 定义两个容器
        confirmed = []
        options = []

        # 2. 遍历检索到的所有商品名的从milvus中的搜索结果
        for item_sea_res in search_result:
            # 2.1 获取extracted_name（LLM提取的商品名）
            llm_extract_item_name = item_sea_res.get('extracted_name')

            # 2.2 获取matches(某一个商品名下的搜索结果)
            item_name_matches = item_sea_res.get('matches')

            # 2.3 (可选) 对某一个商品名下的搜索结果根据分数进行降序排序
            item_name_matches_sorted = sorted(item_name_matches, key=lambda x: x['score'], reverse=True)

            # 2.4 (收集--->分数值大于0.7的搜索结果:高置信度)
            high = [h for h in item_name_matches_sorted if h.get('score') > self._config.item_name_high_confidence]

            # 2.5 高置信度的有
            if high:
                # 1) 是否从向量数据库中搜索到的商品名还和LLM提取到的相等，如果想等，最精准
                extract = next((h for h in high if h.get('item_name') == llm_extract_item_name), None)

                if extract:  # 条件很难触发(有可能 概率低)
                    picked = extract.get('item_name')
                    if picked not in confirmed:  # 去重（认为有重复的 去重）
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    # 有多个分数值比较高的商品名[0.75,0.74,0.71]
                    top_score = high[0]['score']
                    if top_score - high[1]['score'] >= self._config.item_name_score_gap:
                        picked = high[0]['item_name']
                        if picked not in confirmed:
                            confirmed.append(picked)
                    else:
                        for h in high[:self._config.item_name_max_options]:
                            picked = h.get('item_name')
                            if picked not in options and picked not in confirmed:
                                options.append(picked)
            # 不是高置信度，有可能是中等置信，也有可能不是中等置信
            else:
                mid = [m for m in item_name_matches_sorted if
                       m['score'] >= self._config.item_name_mid_confidence
                       and m.get('item_name') not in options
                       and m.get('item_name') not in confirmed]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get('item_name'))  # 一次循环内部只留三个可选的

        return confirmed, options[:self._config.item_name_max_options]  # 最终只留三个可选的商品名给前端

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):

        """
        # RS12数字万用表和RS13数字万用表
        item_name:RS12数字万用表:0.90
        item_name:万用表:0.71
        item_name:RS13数字万用表:0.88
        Args:
            confirmed:
            search_result:[]

        Returns:

        """

        # 1. 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for search_result in search_result:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed
        if not item_name_score:
            return confirmed

        # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if
                max_score - score <= self._config.item_name_score_gap]


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """查询理解 + 实体确认（V5 统一入口节点）。

        一次 LLM 调用产出实体、自包含改写、意图、语义子问题与每个子问的策略标注；
        本节点只负责：
        1. 把历史会话喂给查询理解（指代消解）；
        2. 已选定知识库时直接采用登记商品名（不再让模型改写后的名字影响检索过滤）；
        3. 未选定知识库时用向量对齐把模型提到的商品名归一到库内商品名，
           对齐不到则短路返回澄清话术。
        """

        # 1. 基础参数
        original_query = state.get('original_query')
        session_id = state.get('session_id')
        kb_id = state.get('kb_id', '')

        # 2. 历史对话(mongodb，10 条 ≈ 5 轮)
        history_context = get_recent_messages(session_id=session_id, limit=10)
        formatted_history = [f"角色:{h.get('role', '')},内容:{h.get('text', '')}" for h in history_context]
        formatted_history_str = " ".join(formatted_history)

        # 3. 已选定产品时，登记名直接作为查询理解的已知实体
        kb_item_names = list((get_kb(kb_id) or {}).get('item_names') or []) if kb_id else []

        # 4. 一次调用完成理解与规划
        plan: QueryPlan = understand_query(
            query=original_query,
            history_text=formatted_history_str,
            known_items=kb_item_names,
            max_subs=self.config.max_sub_queries,
            dedup_ratio=self.config.sub_query_dedup_ratio,
        )

        # 5. 实体确认
        if kb_item_names:
            confirmed, options = kb_item_names, []
        elif plan.item_names:
            confirmed, options = self._aligner.search_and_align(plan.item_names, kb_id=kb_id)
        else:
            confirmed, options = [], []

        # 6. 决策并写回执行计划
        self._decide(confirmed, options, state, plan)

        state['history'] = history_context
        return state

    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState, plan: QueryPlan):
        """写回查询计划；商品名不明确或识别不到时短路返回提示，不进入检索。"""

        state['rewritten_query'] = plan.rewritten_query
        state['item_names'] = confirmed or list(plan.item_names or [])
        state['intent'] = plan.intent
        state['sub_queries'] = [sub.query for sub in plan.subs]
        state['query_plan'] = [asdict(sub) for sub in plan.subs]
        # 短路也要留空结构，避免下游按下标取值时越界
        state['evidence_per_q'] = []
        state['answer_mode_per_q'] = []

        if confirmed:
            self.logger.info(f"商品名已确认: {confirmed}，子问 {len(plan.subs)} 个")
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
            self.logger.info(f"商品名不精确，返回候选给用户确认: {options}")
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            self.logger.info(f"未识别到任何商品名, 查询理解提取结果: {plan.item_names}")


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)
