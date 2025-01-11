# Financial Chat Bot 

* This is a financial chat application built using Streamlit. It integrates advanced Retrieval-Augmented Generation (RAG) techniques to provide meaningful responses to user queries. The application includes features for user authentication, session management, and query history tracking.

- A RAG(Retrieval-Augmented Generation) application will answer the questions regarding the Financial details of Three Industries(Infosys, Uber and Alteryx). 


### Prerequisites

- Python 3.12.7

- Streamlit

- Llama Index

- ChromaDB

- OpenAI API Key

- LlamaParser API Key

- Groq API Key

- Gemini API Key


## setup

1. Clone the Reporistory
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create an Environment
```bash
conda create -n env_name python=3.12.7 -y
```

3. Install Dependencies:
```bash
conda install -c conda-forge chromadb 
pip install -r requirements.txt
```

4. SetUp environment variables:
```bash
OPENAI_API_KEY=<your_openai_api_key>
LLAMAPARSER_API_KEY=<your_llamaparser_api_key>
GROQ_API_KEY=<your_groq_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
PINECONE_API_KEY = <your_pinecone_api_key>
```

5. Run App:
```bash
streamlit run app.py
```


## Useage:
- Login and Signup

- Signup: Create a new account using a valid email and password.

- Login: Use your credentials to access the dashboard.

- Dashboard

- View your session history and interact with the RAG-based chat.

- Select or start a new session from the sidebar.

- Enter queries, and the system will respond using the selected model.

- Sidebar Options

- Select an LLM model: Gpt-4o, Llama3-70b, or Gemini-1.5-flash.

- View and manage sessions.

- Logout securely.


## Code Overview

#### Main Files:
- app.py: Entry point of the application; handles navigation and authentication.

- login_signup.py: Implements login, signup, and dashboard logic.

- rag_app3.py: Manages RAG-specific logic, including query rewriting and LLM interaction.

- original_rag.py: Core functions for LLM initialization, context retrieval, and query handling.

#### Key Functions:

- get_engine(): Initializes the query engine with ChromaDB.

- rewrite_query(): Modifies user queries based on session history.

- get_context(): Retrieves relevant documents for a query.

- get_combined_context(): Combines query context with session history.


## Overview:

- RAG Application with basic features that able to save the sessions and chat history in json formatted file(history.json) and login credentials in json formatted file(credentials.json).
- Both files are initialised and space for the user is alloted at the time of signup.

```bash
Credentials.json:
{
    "mail": "password",
}
```

```bash
History.json:
{
    "mail": [
        [
            "sission-history-1",            # this will include all the query+"\n"+response of 1st session
            "you: Query-1",
            "AI: Response-1",...
        ],
        [
            "sission-history-2",            # this will include all the query+"\n"+response of 2nd session
            "you: Query-1",
            "AI: Response-1",...
        ],...
    ]
}
```


### Workflow
```bash
streamlit app
    |
    | 
    |--app.py
          |----login_signin.py
                    |----rag_app.py
                            |----original_rag.py
```


## Architecture:

1. User signup
2. User Login
3. Select Model[optional] - By default Openai Gpt-4o
4. Enter Query and submit
5. If history exists query will rewrittened and passed to the query engine to retrieve relavent documents.
6. If history existory exist for the particular session, than history will be included into the context.
7. query will re-emitted with the prompt template and passed to the selected model
8. result save into the history.

![Image](Architecture2.png "icon")


## Frontend

- We can able to select models and sessions and chat history is displayed as per the selected session

![image.png](screen.png "icon")



## Features

- User Authentication: Login and Signup functionality with secure credentials storage.
- Session Management: Maintain multiple sessions for each user.
- RAG-based Query System: Uses advanced LLMs (e.g., GPT-4, LLaMA, Gemini) for generating responses.
- History Tracking: Stores and displays previous interactions for better context.
