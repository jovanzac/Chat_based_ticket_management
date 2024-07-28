import os

from llama_index.core import Settings
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer

from redisvl.schema import IndexSchema

from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine.types import ChatMode


from src.settings import Config
from llama_index.core.query_engine import RetrieverQueryEngine


class RedisStore :
    def __init__(self, index_name, index_prefix) :
        # Initialize the vector store
        self.vector_store = RedisVectorStore(
            schema=self.get_custom_schema(index_name, index_prefix),
            redis_url="redis://localhost:6379",
        )
        
        
        # Set up the ingestion cache layer
        self.cache = IngestionCache(
            cache=RedisCache.from_host_and_port("localhost", 6379),
            collection=f"redis_cache_{index_name}",
        )
        
        # Initialize the document store
        self.docstore = RedisDocumentStore.from_host_and_port(
            "localhost",
            6379,
            namespace=f"document_store_{index_name}"
        )
        
        self.chat_store = RedisChatStore(
            redis_url="redis://localhost:6379", 
            ttl=300,
        )

    
    def get_custom_schema(
            self, 
            index_name = "rough_index", 
            index_prefix = "doc_rough"
        ) :
        """Return the custom schema for the index
        """
        custom_schema = IndexSchema.from_dict(
            {
                "index": {"name": index_name, "prefix": index_prefix},
                # customize fields that are indexed
                "fields": [
                    # required fields for llamaindex
                    {"type": "tag", "name": "id"},
                    {"type": "tag", "name": "doc_id"},
                    {"type": "text", "name": "text"},
                    # custom vector field for bge-small-en-v1.5 embeddings
                    {
                        "type": "vector",
                        "name": "vector",
                        "attrs": {
                            "dims": 384,
                            "algorithm": "hnsw",
                            "distance_metric": "cosine",
                        },
                    },
                ],
            }
        )
        
        return custom_schema
    
    
    def add_embedded_nodes(self, nodes, docstore=False) :
        self.vector_store.add(nodes)
        if docstore :
            self.docstore.add_documents(nodes)

    
    
class DataIngestion :
    def __init__(self, config, datastorage) :
        self.config = config
        self.store = datastorage
    
    def create_documents_from_files(self, loc="./data") :
        documents = None
        if os.path.exists(loc) :
            documents = SimpleDirectoryReader(
                input_dir=loc,
                file_extractor={
                    ".pdf": self.config.pdf_parser,
                },
                recursive=True
            ).load_data()
            
            
        return documents
    
    
    def create_nodes(self, documents) :
        nodes = self.config.node_parser.get_nodes_from_documents(documents)
        
        return nodes
    
    
    def storage_context(self) :
        storage_context = StorageContext.from_defaults(
            docstore=self.store.docstore,
            vector_store=self.store.vector_store,
        )
        
        return storage_context
    
    
    def ingestion_pipeline(self, nodes) :
        pipeline = IngestionPipeline(
            transformations=[
                self.config.embedding_model,
            ],
            docstore=self.store.docstore,
            vector_store=self.store.vector_store,
            cache=self.store.cache,
        )
        
        pipeline.run(nodes=nodes)
        
    
    def create_chat_memory(self) :
        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=self.store.chat_store,
            chat_store_key="OptyVergeUser1",
        )
        
        return chat_memory
    
    
    def create_vector_index(self, storage_context) :
        # Create the vector store index
        index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            embed_model=self.config.embedding_model,
        )
        
        return index
    
    
if __name__ == "__main__" :
    print("COnfig and Setup...")
    redis = RedisStore("optyverge_support", "opty")
    
    config = Config()
    Settings.llm = config.base_llm
    res = config.base_llm.complete("What is the capital of France?")
    print(res.text)
    
    ingestion = DataIngestion(config, redis)
    
    print("Getting documents and nodes...")
    # documents = ingestion.create_documents_from_files(loc="./data")
    # print(f"len(docs): {len(documents)}")
    # nodes = ingestion.create_nodes(documents)
    # print(f"len(nodes): {len(nodes)}")
    storage_context = ingestion.storage_context()
    print("Setting up and starting the ingestion pipeline...")
    # ingestion.ingestion_pipeline(nodes)
    print("Done with Ingestion..")
    print("Creating vector index")
    index = ingestion.create_vector_index(storage_context)
    
    retriever = index.as_retriever(similarity_top_k=2)
    
    query_str = input(">>")
    nodes = retriever.retrieve(query_str)
    
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=config.base_llm)
    response = query_engine.query(query_str)
    
    
    print("Response without Prompt Template: ")
    print(response)
    
    """With prompt template and memory
    """
    
    qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, answer the query."
    "You are a friendly chatbot so make sure to be helpful."
    "Also, always respond in complete sentences\n"
    "If a user mentions an issue, ask if they want to create a ticket for it.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    # Text QA Prompt
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                """Always answer the question, even if the context isn't helpful.
                You are a support bot to help out users in a ticket management system.
                Help users with their queries and generate tickets for their queries 
                if requested. End every response with a thank you.
                """
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=qa_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    
    memory = ingestion.create_chat_memory()
    
    prompt_query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=text_qa_template,
        llm=config.base_llm,
    )
    
    chat_query_engine = index.as_chat_engine(
        # chat_mode="context",
        chat_mode=ChatMode.CONTEXT,
        context_prompt=qa_prompt_str,
        system_prompt="""Always answer the question, even if the context isn't helpful.
                You are a support bot to help out users in a ticket management system.
                Help users with their queries and generate tickets for their queries 
                if requested. End every response with a thank you.
                """,
        memory=memory,
    )
    
    response = prompt_query_engine.query(query_str)
    
    print("Response after applying template...")
    print(response)
    
    response = chat_query_engine.chat(query_str)
    
    print("Response after applying memory...")
    text = response.response
    print(type(text), text)