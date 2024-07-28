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
                "Created_at": ticket["Created_at"],
                "Updated_at": ticket["Updated_at"],
            }
        )
        
        
if __name__ == "__main__" :
    db_manager = DatabaseManager()