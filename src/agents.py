from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine.types import ChatMode

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class QueryAgent :
    def __init__(self, config) :
        self.config = config
        
        self.lang_model = ChatGroq(
            model_name="gemma2-9b-it",
            temperature=1,
            max_tokens=1024,
        )
        self.classification_llm = self.class_prompt_structure()
        
        
    def simple_retrieval(self, query_str, index, similarity=2) :
        retriever = index.as_retriever(similarity_top_k=similarity)
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever, 
            llm=self.config.base_llm
        )
        response = query_engine.query(query_str)
        
        return response
    
    
    def template_based_retrieval(self, query_str, index, similarity=2) :
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
        
        response = prompt_query_engine.query(query_str)
        
        return response
    
    
    def memory_based_retrieval_with_template(self, query_str, index, memory, similarity=2) :
        # retriever = index.as_retriever(similarity_top_k=similarity)
        
        qa_prompt_str = """Context information is below.\n"
            ---------------------\n"
            {context_str}\n"
            ---------------------\n"
            Given the context information, answer the query."
            You are a friendly chatbot so make sure to be helpful."
            Also, always respond in complete sentences\n"
            If a user mentions an issue, ask if they want to create a ticket for it.\n"
            Query: {query_str}\n
            "Answer: """
        
        chat_query_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            context_prompt=qa_prompt_str,
            system_prompt="""Always answer the question, even if the context isn't helpful.
                    You are a support bot to help out users in a ticket management system.
                    Help users with their queries and generate tickets for their queries 
                    if requested. DO NOT generate a response if it is not relevant to ticketing or 
                    any user query on the companies' products.
                    When generating a a ticket, follow the given format strictly:
                        Ticket ID: # Generate a unique ticket ID
                        Subject: # Generate an insightful subject
                        Description: # Describe in detail the issue
                        Date: # Generate a date in the format YYYY-MM-DD
                        Status: # Include a status of open, in review, or closed
                    """,
            memory=memory
        )
        
        response = chat_query_engine.chat(query_str)
        
        return response
    
    
    def class_prompt_structure(self) :
        classification_template = PromptTemplate.from_template(
            """You're job is to classify queries. Given a user question below, classify it
            as belonging to either "Get_Time", "Get_Stock_Price", "Send_Whatsapp_Message" or "Miscellaneous". 

            <If the user query is about the current time in any country, then classify the question as "Get_Time".
            The response should be of the format "Get_Time,IANA" where IANA is a valid time zone from
            the IANA Time Zone Database like "America/New_York" or "Europe/London". If the country isn't mentioned
            assume it to be India>
            
            <If the user query is about the stock price of a share of any company in the market, then classify the question
            as "Get_Stock_Price". The Respose should be of the format "Get_Stock_Price,Symbol" where Symbol
            is the Ticker symbol of the concerned company.>
            
            <If the user query is about sending a whatsapp message to a particular user, then classify the question as "Send_Whatsapp_Message". 
            The response should be in the format "Send_Whatsapp_Message,Name,Message" where Name is the name of the person mentioned and
            Message is the message to be sent that is taken from the user's query word for word.>
            
            <If the user query is about any other subject or topic, classify the question as "Miscellaneous">

            <question>
            {question}
            </question>

            Classification:
            """
            )

        classification_chain = (
            classification_template
            | self.lang_model
            | StrOutputParser()
        )
        
        return classification_chain
    
    
    def route(self, query) :
        print("Here in route")
        route_class = self.classification_llm.invoke({"question": query}).split(",")
        if route_class[0] == "Get_Time" :
            print("here in get time")
            return "The time is: " + self.get_time(route_class[1].strip())

        elif route_class[0] == "Get_Stock_Price" :
            response = self.get_stock_price(route_class[1].strip())
            return f"The price of {response['name']}'s stock is currently USD {response['price']}"
        
        elif route_class[0] == "Send_Whatsapp_Message" :
            print("Here")
            name = route_class[1].strip().capitalize()
            number = self.phone_book[name]
            print(f"number: {number}")
            mssg = route_class[2].strip()
            self.send_whatsapp_mssg(mssg, number)
            
            return f"Message {mssg} sent successfully to {name}"

        else :
            return self.invoke_rag(query)