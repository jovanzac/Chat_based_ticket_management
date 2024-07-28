## Ticketing Management System Simplified!

- Implements a powerful and custom ReAct agent to simplify the ticket creation and management process for clients, users and all other stakeholders.
- Implements advanced RAG concepts to enable the chatbot to accurately respond to user queries regarding the companies' products. 
- Prompt engineering to better serve the users
- Uses Redis as the vector store, document store and for caching. Also used to store and retrieve conversation history for better user experince.
- Mongodb to enable storing and retrieving all created tickets


## Setup and run
- Step1: Install redis and start a server on localhost port 6379(default for redis)
Follow [steps here](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)

- Step2: install requirements 
```
%pip install -r requirements.txt
```

- Step3: Add environment variables in a file named .env
```
GROQ_API_KEY = <Your API key>
URI = <MongoDb URI>
```

- Step3: Start streamlit
```
streamlit run streamlit_app.py
```