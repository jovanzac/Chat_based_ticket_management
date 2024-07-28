import re
from datetime import datetime

from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine.types import ChatMode

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.database import DatabaseManager


class ReactAgent :
    def __init__(self, config, db_manager) :
        self.config = config
        
        self.lang_model = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=1,
            max_tokens=1024,
        )
        self.classification_llm = self.class_prompt_structure()
        
        self.db_manager = db_manager
        
        
    def llama_templates(self, tpl_type, tickets=None) :
        if tpl_type == "regular" :
            return """Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Given the context information, answer the query.
            You are a friendly chatbot so make sure to be helpful.
            Also, always respond in complete sentences\n
            If a user mentions an issue, ask if they want to create a ticket for it.\n
            Query: {query_str}\n
            Answer: 
            """
                
        elif tpl_type == "regular_sys" :
            return """Always answer the question, even if the context isn't helpful.
            You are a support bot to help out users in a ticket management system.
            Help users with their queries and generate tickets for their queries 
            if requested. DO NOT generate a response if it is not relevant to ticketing or 
            any user query on the companies' products.
            """
                
        elif tpl_type == "classification" :
            return """You're job is to classify queries. Given a user question below, classify it
            as belonging to either "Create_Ticket", "Retrieve_Ticket", or "Generic". 

            <If the user describes an issue and asks to have a ticket created for it, then classify the question as "Create_Ticket". The response should be in the format "Create_Ticket" and nothing else must be generated.>
            
            <If the user asks to retrieve all previously created tickets or a specific ticket, classify the question as "Retrieve_Ticket".>
            
            <If the user query is about anything else, such as describing an issue the user has (without asking to create a ticket), or asking for some other information, classify the question as "Generic".>

            <question>
            {question}
            </question>

            Classification:
            """
            
        elif tpl_type == "create_ticket_sys" :
            return""" Your purpose is to create tickets to customer queries and issues. You are not a chat agent but 
            simply create tickets. These created tickets must be your only response to the query. Tickets created
            must strictly follow the given format:
            
            ```
            Ticket ID: <Ticket_ID>,\n
            Subject: <Subject>,\n
            Description: <Description>,\n
            Status: <Status>,\n
            Priority: <Priority>,\n
            Created at: <Created_at>\n
            ```
            
            Here, Ticket_ID is a random unique ID to be generated for the ticket; Subject and Description must be 
            created from the context of the conversation with the user; Status is assigned "in review" by default;
            Priority is assigned "medium" by default unless specified otherwise by the user; Created_at is the time 
            when the ticket was created and will be provided with the prompt.
            """
            
        elif tpl_type == "create_ticket" :
            return """Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Given the context information, create a ticket for the user. Follow strictly the 
            format specified for creating the ticket.

            Query: {query_str}\n
            Answer: 
            """
            
        elif tpl_type == "retrieve_ticket" :
            tickets_template = f"""Context information is below.\n
            ---------------------\n
            {tickets}\n
            ---------------------\n
            """
            
            prompt_tmpl = tickets_template + """
            The provided context lists all the tickets that have been retrieved from the database. Format the tickets
            appropriately and list them out for the user\n

            Query: {query_str}\n
            Answer: 
            """
            
            return prompt_tmpl
        
        
    def class_prompt_structure(self) :
        classification_template = PromptTemplate.from_template(
            self.llama_templates(tpl_type="classification")
        )

        classification_chain = (
            classification_template
            | self.lang_model
            | StrOutputParser()
        )

        return classification_chain
    
    
    def route(
            self, 
            query,
            index=None, 
            memory=None, 
            similarity=2
        ) :
        print("Here in route")
        route_class = self.classification_llm.invoke({"question": query})
        print(f"Selected route is: {route_class}")
        if re.search("Create_Ticket", route_class, re.IGNORECASE) :
            current_datetime = datetime.now()
            query_str = f"{query}\n\n Current datetime: {current_datetime}"
            response = self.create_ticket_mbrt(
                query_str=query_str,
                index=index,
                memory=memory,
                similarity=similarity
            )
            
            lines = response.response.strip().split(",\n")

            # Extract the data
            data = {}
            for line in lines:
                key, value = line.split(": ", 1)
                data[key.strip()] = value.strip()

            ticket_info = list(data.values())
            self.db_manager.insert_ticket(
                {
                    "Ticket_ID": ticket_info[0],
                    "Subject": ticket_info[1],
                    "Description": ticket_info[2],
                    "Status": ticket_info[3],
                    "Priority": ticket_info[4],
                    "Created_at": ticket_info[5]
                }
            )
            
            return f"Your ticket has been created:\n {response}"
        
        elif re.search("Retrieve_Ticket", route_class, re.IGNORECASE) :
            all_tickets = self.db_manager.retrieve_all_docs()
            
            return self.retrieve_tickets_mbrt(
                query_str=query,
                index=index,
                memory=memory,
                tickets=all_tickets,
                similarity=similarity
            )
            
            
            
        
        else :
            return self.simple_mbrt(
                query_str=query,
                index=index,
                memory=memory,
                similarity=similarity
            )
        
        
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
        
        qa_prompt_str = self.llama_templates(tpl_type="regular")

        # Text QA Prompt
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self.llama_templates(
                    tpl_type="regular_sys"
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
    
    
    def simple_mbrt(self, query_str, index, memory, similarity=2) :
        """
        MBRT: Memory-Based Retrieval with Template
        """
        qa_prompt_str = self.llama_templates(tpl_type="regular")
        
        chat_query_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            context_prompt=qa_prompt_str,
            system_prompt=self.llama_templates(
                tpl_type="regular_sys"
            ),
            memory=memory,
            similarity_top_k=similarity,
        )
        
        response = chat_query_engine.chat(query_str)
        
        return response
    
    
    def create_ticket_mbrt(self, query_str, index, memory, similarity=2) :
        """
        Used when the user wants to create a ticket
        MBRT: Memory-Based Retrieval with Template
        """
        qa_prompt_str = self.llama_templates(tpl_type="create_ticket")
        
        chat_query_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            context_prompt=qa_prompt_str,
            system_prompt=self.llama_templates(
                tpl_type="create_ticket_sys"
            ),
            memory=memory,
            similarity_top_k=similarity,
        )
        
        response = chat_query_engine.chat(query_str)
        
        return response
    
    
    def retrieve_tickets_mbrt(self, query_str, index, memory, tickets, similarity=2) :
        """
        Used when the user wants to retrieve tickets
        MBRT: Memory-Based Retrieval with Template
        """
        qa_prompt_str = self.llama_templates(tpl_type="retrieve_ticket", tickets=tickets)
        
        chat_query_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            context_prompt=qa_prompt_str,
            system_prompt=self.llama_templates(
                tpl_type="create_ticket_sys"
            ),
            memory=memory,
            similarity_top_k=similarity,
        )
        
        response = chat_query_engine.chat(query_str)
        
        return response
    