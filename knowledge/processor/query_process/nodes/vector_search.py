import json
from typing import Union, Dict, Any, Tuple, List

from torch.nn.functional import embedding

from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import milvus_client, item_names_filter, create_hybrid_search_requests, \
    execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    """向量检索"""
    name = "search_embedding"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query,validate_item_names=self._validate_state(state)
        # 2. 获取嵌入模型
        try:
            embedding_model=AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state

        # 3. 获取milvus客户端
        try:
            milvus_client=StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return state

        # 4. 对问题嵌入
        try:
            embed_query=generate_bge_m3_hybrid_vectors(model=embedding_model,embedding_documents=[validated_query])
        except Exception as e:
            self.logger.error(f"问题{validated_query}嵌入失败")
            return state
        # 5. 构建过滤表达式以及表达式参数
        filter_expr, filter_expr_param = item_names_filter(validate_item_names)
        try:
            # 6. 创建混合请求
            hybrid_search_request = create_hybrid_search_requests(
                dense_vector=embed_query['dense'][0],
                sparse_vector=embed_query['sparse'][0],
                expr=filter_expr,
                expr_params=filter_expr_param)

            # 7. 执行混合搜索请求
            hybrid_search_reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_request,
                output_fields=['chunk_id', 'content', 'item_name', 'title'])

            # 8. 获取搜索结果
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state

            # 9. 更新state 返回
            return {
                "embedding_chunks": hybrid_search_reps[0]
            }
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return state


    def _validate_state(self,state:QueryGraphState)->Tuple[str, List[str]]:
        rewritten_query=state.get('rewritten_query')
        item_names=state.get('item_names')

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name,field_name='rewritten_query',expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name,field_name='item_names',expected_type=list)
        return rewritten_query,item_names

if __name__ == '__main__':
    state = {
        "original_query": "万用表的使用如何测量电阻",
        "rewritten_query": "RS-12 数字万用表的使用如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))