from src.settings import Config
from src.agents import QueryAgent
from src.docs_ingestion import DataIngestion, RedisStore


config = Config()
query_agent = QueryAgent(config)
redis_store = RedisStore("optyverge_support", "opty")
data_ingestion = DataIngestion(config, redis_store)


def get_response(query_str) :
    storage_context = data_ingestion.storage_context()
    index = data_ingestion.create_vector_index(storage_context)
    memory = data_ingestion.create_chat_memory()
    
    response = query_agent.memory_based_retrieval_with_template(
        query_str,
        index,
        memory,
        similarity=2
    )
    
    return response

