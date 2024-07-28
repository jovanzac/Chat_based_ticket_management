import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager :
    URI = os.environ.get("URI")
    
    def __init__(self, ) :
        self.client = MongoClient(self.URI, server_api=ServerApi('1'))
        print("Connection established")
        self.db = self.client["tickets"]
        
    
    def insert_ticket(self, ticket, username="DefaultUser") :
        user_coll = self.db[username]
        
        user_coll.insert_one(
            {
                "Ticket_ID": ticket["Ticket_ID"],
                "Subject": ticket["Subject"],
                "Description": ticket["Description"],
                "Status": ticket["Status"],
                "Priority": ticket["Priority"],
                "Created_at": ticket["Created_at"]
            }
        )
        
    
    def retrieve_all_docs(self, username="DefaultUser") :
        user_coll = self.db[username]
        documents = user_coll.find()
        
        return documents
    
        
if __name__ == "__main__" :
    db_manager = DatabaseManager()
    
    db_manager.insert_ticket(
        ticket={
            "Ticket_ID": "1234",
            "Subject": "Ticket 1",
            "Description": "Ticket 1 description",
            "Status": "In review",
            "Priority": "Medium",
            "Created_at": "2022-01-01"
        }
    )