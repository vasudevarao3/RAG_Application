from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq
import google.generativeai as genai
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import time
import streamlit as st


from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

LLAMAPARSER_API_KEY = "llx-iDZxH9vPYjowBAVbLOfMvnt95SJdgOfhKZTMjmbL8mba5R1D"    #Linga


loader = LlamaParse(
        api_key=LLAMAPARSER_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        verbose=True,
    )

## Ensure OpenAI API Key is available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLAMAPARSER_API_KEY = os.environ.get("LLAMAPARSER_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

if not LLAMAPARSER_API_KEY:
    raise ValueError("LLamaParser API key is missing. Ensure it is set in the .env file.")

if not GROQ_API_KEY:
    raise ValueError("Groq API key is missing. Ensure it is set in the .env file.")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key is missing. Ensure it is set in the .env file.")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Ensure it is set in the .env file.")


def get_engine(): 
    # Define the directory containing the text file and the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # input_directory = os.path.join(current_dir, "pdfs")
    input_files = ["./pdfs/alteryx-2023.pdf", "./pdfs/gitlab-2023.pdf", "./pdfs/infosys-2023.pdf", "./pdfs/uber-2023.pdf"
                   "./pdfs/alteryx-2022.pdf", "./pdfs/gitlab-2022.pdf", "./pdfs/infosys-2022.pdf", "./pdfs/uber-2022.pdf",
                   "./pdfs/alteryx-2021.pdf", "./pdfs/gitlab-2021.pdf", "./pdfs/infosys-2021.pdf", "./pdfs/uber-2021.pdf",
                   "./pdfs/alteryx-2020.pdf", "./pdfs/infosys-2020.pdf", "./pdfs/uber-2020.pdf"]

    persistent_directory = os.path.join(current_dir, "db", "chroma_db_llama_parser_file_size")
    print(persistent_directory)


    # Use OpenAI embeddings for vectorization
    embedding_model = OpenAIEmbedding(model = "text-embedding-3-small")
    Settings.embed_model = embedding_model

    # Initialize connection to Pinecone
    pc = PineconeGRPC(api_key =  PINECONE_API_KEY)
    index_name = "capston-llama-parser"
    
    # Create your index (can skip this step if your index already exists)
    existing_indexes=[index_info["name"] for index_info in pc.list_indexes()]
    print(existing_indexes)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        parser = LlamaParse(
            api_key=LLAMAPARSER_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
        )


        # Document Loading
        print("Uploading Documents, Please wait...")
        documents = parser.load_data(file_path = input_files)
        print("Documents successfully loaded")

        # Defining Splitter
        print("Splitting documents into chunks...")
        splitter = SentenceSplitter(
            include_metadata=True
        )
        
        nodes = splitter.get_nodes_from_documents(documents)
        for node in nodes:
            if node.text!= "":
                node.embedding = embedding_model.get_text_embedding(node.get_content(metadata_mode="all"))


        pinecone_index = pc.Index(index_name)
        print(pinecone_index, index_name)

        # See that it is empty
        print(pc.Index(index_name).describe_index_stats())
        print("\n")

        # Initialize VectorStore
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        vector_store.add(nodes = nodes)

        # After adding nodes
        print(pc.Index(index_name).describe_index_stats())
        
    else:
        print("Vector DB exist...")
        pinecone_index = pc.Index(index_name)
        print(pinecone_index, index_name)

        # See that it is empty
        print(pc.Index(index_name).describe_index_stats())
        print("\n")

        # Initialize VectorStore
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        # print(vector_store)
        print("Vector Store loaded")

        # Instantiate VectorStoreIndex object from your vector_store object
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


    # Create Retriever and Query Engine with SimilarityPostprocessor
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=15)
    print("Retriever")
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
    print("Post Processor")
    query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])
    print("Query Engine")

    return query_engine
    

# rewrite the query according to the history
def rewrite_query(query, history =""):

    llm = OpenAI(model="gpt-3.5-turbo")

    query_prompt = f"""
        "The original query is as follows: {query}\n"
        "We have provided an existing history: {history}\n"
        "We have the opportunity to rewrite the original query "

        - Rewrite the original query into a clear , specific suitable for retrieving relevant information from a vector database.
        - Keep in mind that your rewritten query will be sent to a query engine of vector database, which does similarity search for retrieving documents.
        - Don't change the meaning of the query.
        - If the history isn't useful or exist, return the original answer.
        - If the query is not clear, use latest history data to relate and rewrite query.
    """
        
    print("*"*50,"Query Prompt","*"*50)
    print(query_prompt)


    chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=  query_prompt
                ),
            ChatMessage(role=MessageRole.USER, content=query),
    ]

    # Creating Chat Prompt Template
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    # print("Creating ChatPromptTemplate:", text_qa_template)

    #QA Response from formatted msgs
    response = text_qa_template.format_messages(chat_text_qa_msgs)
    # print("*"*50,"QA Response: ", "*"*50,"\n", qa_response)

    print(response)

    rewritter = llm.chat(response)

    rewritten_query = rewritter.message.blocks[0].text
    # rewritten_query = llm.complete(query_prompt)
                      
    print("*"*50,"Re Written Prompt", "*"*50)
    print(rewritten_query)
    return rewritten_query

# Get relavent document documents according to the query
def get_context(query_engine, combined_query):

    print("query_engine")
    results=query_engine.query(combined_query)
    print("*"*50,"Query Response", "*"*50)
    print(results)
    pprint_response(results,show_source=True)
    # print("*"*50,"Re Source Nodes of Result", "*"*50,"\n", "Before str - ",results.source_nodes)

    context_str = ""

    for index, doc in enumerate(results.source_nodes):
        # context_list.append(doc.text)
        context_str = context_str + "\n" + doc.text
        print(index, doc.text)
        print(len(context_str))

    return context_str


#Sends the selected LLM model
def get_model(model_name):
    if model_name == "Gpt-4o":
        # Initialize openai model
        llm = OpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)
        print("OpenAI model is Created")

    if model_name == "Llama3-70b":
        # Initialize Groq model
        llm =  Groq(model="llama3-70b-8192", temperature=0.1, api_key=GROQ_API_KEY)
        print("Llama3 is Created")
    
    if model_name == "Gemini-1.5-flash":
        # Initialize Gemini-1.5-flash model
        genai.configure(api_key=GEMINI_API_KEY)
        llm = genai.GenerativeModel(model_name = "gemini-1.5-flash")
        print("Gemini is Created")

    return llm
    

# Prmopt template with query
def get_prompt(history, context, query):
    #creating prompt string 
    qa_prompt = f"""
            You are an AI assisstent[you are aware of Infosys, Uber and Alteryx companies financial reports] that answers questions strictly based on the provided Relevants Documents and conversation history, if available.
            If no context is available or no context is used to answer, respond with 'No relevant information found.'

            ### Conversation History:
            {history}

            ### Relevant Documents:
            {context}

            ### User Query:
            {query}

            ### Guidelines for Answer:
            1. Provide the **Final Answer** concisely without displaying the step-by-step reasoning process or irrelevant details.
            2. If the user query does not specify a year, provide data for all available years mentioned in the context.
            3. If the user query does not specify a industry name, provide data for all available industries mentioned in the context 
            4. Include an **Explanation** of how the relevant documents helped in answering the query, focusing on why the context is relevant.
            5. If no relevant documents are found, explicitly state: 'No relevant information found.'
            6. If query context is out of the box, just say sorry and I am not aware about it.
        """
    # print(qa_prompt)

    print("*"*50,"QA Prompt: ", "*"*50,"\n", qa_prompt)

    # Chat Text QA msgs
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Always answer the question, even if the context isn't helpful."
            ),
        ChatMessage(role=MessageRole.USER, content=qa_prompt),
    ]

    # Creating Chat Prompt Template
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    # print("Creating ChatPromptTemplate:", text_qa_template)

    #QA Response from formatted msgs
    qa_response = text_qa_template.format_messages(chat_text_qa_msgs)
    # print("*"*50,"QA Response: ", "*"*50,"\n", qa_response)

    return qa_response


def create_db(uploaded_files, file_details):
    print(file_details)       #  {'filename': 'CAG_Research_Paper.pdf', 'type': 'application/pdf', 'size': 121164}
    print(uploaded_files)
    # UploadedFile(file_id='c7c5689c-4086-4fa2-9277-880d4d58f8be', name='CAG_Research_Paper.pdf', type='application/pdf', size=121164, _file_urls=file_id: "c7c5689c-4086-4fa2-9277-880d4d58f8be"
    # upload_url: "/_stcore/upload_file/d09265e4-025c-4f92-a8a4-abcbfddc37d4/c7c5689c-4086-4fa2-9277-880d4d58f8be"
    # delete_url: "/_stcore/upload_file/d09265e4-025c-4f92-a8a4-abcbfddc37d4/c7c5689c-4086-4fa2-9277-880d4d58f8be"
    # )

    # if "" not in st.session_state:
    #     st.session_state["authenticated"] = False
    #     st.session_state["user"] = None
