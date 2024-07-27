import os

from llama_index.readers.file import PyMuPDFReader

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv
load_dotenv()


class Config :
    def __init__(self) :
        os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
        
        self.base_llm = Groq(
            model="gemma2-9b-it",
            temperature=1,
            max_tokens=1024,
        )
        
        Settings.llm = self.base_llm
        
        self.embedding_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        self.node_parser = SentenceSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )
        
        self.pdf_parser = PyMuPDFReader()