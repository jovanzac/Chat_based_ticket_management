from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


class QueryAgent :
    def __init__(self, config) :
        self.config = config
        
        
    def simple_retrieval_query(self, query_str, index, similarity=2) :
        retriever = index.as_retriever(similarity_top_k=similarity)
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever, 
            llm=self.config.base_llm
        )
        response = query_engine.query(query_str)
        
        return response
    
    
    def retrieval_template_query(self, query, index, similarity=2) :
        retriever = index.as_retriever(similarity_top_k=similarity)
        
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
        
        prompt_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=text_qa_template,
            llm=self.config.base_llm,
        )