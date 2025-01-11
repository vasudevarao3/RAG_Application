# This is Dashpage
import streamlit as st
import json
from original_rag_pinecone import get_engine, get_context, get_model, get_prompt, rewrite_query, create_db

HISTORY_FILE = "history.json"

# chat_index to select the index of particular session in the history
chat_index=0


def process_file(uploaded_files):
    # Placeholder function to handle the uploaded file
    file_details = []
    for file in uploaded_files:
        file_details.append({
            "filename": file.name,
            "type": file.type,
            "size": file.size  
        })
    st.write("File uploading...")
    st.json(file_details)
    create_db(uploaded_files, file_details)
    st.write("File uploaded successfully!")


#Display history, Query followed by AI Reponse from the history, when history got updated
def display_history(email, history):        
    for index in range(1,len(history[email][chat_index]),2):
        st.write("You: ", history[email][chat_index][index])
        if index+1<len(history[email][chat_index]):
            st.write("AI: ", history[email][chat_index][index+1])


#Implementing Rag dashboard with side bar
def dash_page_rag(email, history):
    global chat_index
    session_history =[]

    # Sidebar setup
    with st.sidebar:
        st.title("Financial Chat Application")

        uploaded_files = st.file_uploader("Upload a file", type=["docx", "pdf", "txt"], accept_multiple_files= True)
        if uploaded_files:
            process_file(uploaded_files)


        # Model selection
        model = st.selectbox("Select Model", ["Gpt-4o", "Llama3-70b", "Gemini-1.5-flash"])

        st.write("-"*50)

        #when new button is clicked, New list is appended to the history of user, only when all the empty sessions are filled
        #file is updated
        if st.button("+ New Chat") and history[email][-1] != [""]:
            history[email].append([""]) 
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f)

        # creating session history list
        for index, chat in enumerate(history[email]):
            # print("session Write", history[email][1])
            if len(chat)>2:
                session_history.append(chat[1])
            else: 
                session_history.append("New Chat")

        # st.write(chat_index, session_history)

        # Radio button to select an session
        selected_item = st.radio("Sessions History: ", session_history)

        # Get the index of the selected session
        chat_index = session_history.index(selected_item)
        

        # Logout button
        st.write("-"*50)
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()

    #Gretting User with Email
    st.subheader(f"*Welcome, {email}!*")
    # st.write(history[email], chat_index)

    #displaing history if exist as per the session with the help of chat_index
    if len(history[email]) and len(history[email][chat_index]):
        display_history(email, history)

    # User input at the bottom
    query = st.text_input("Enter Your Query:")
    submit = st.button("Submit")

    #if button clicked
    if submit and query.strip():

        query_engine = get_engine() 
        print("Query Engine")

        current_query_history = []
        current_answer_history =[]
        #if session contains history than query will be rewritten according to the chat session history
        for index in range(1, len(history[email][chat_index]), 2):
            current_query_history.append(history[email][chat_index][index])
            current_answer_history.append(history[email][chat_index][index+1])

        rewritten_query = rewrite_query(query, current_query_history)
        # rewritten_query = query
        print("Rewritten Query: ", rewritten_query)
        st.write("Rewritten - ", rewritten_query)

        #getting relavent documents in string formate according to the query
        context = get_context(query_engine, rewritten_query)
        print("Context")
        # st.write("context: ", context)

        #model initialization
        llm = get_model(model)

        #query with prompt template
        if len(current_answer_history):
            query_with_prompt_template = get_prompt(current_answer_history, context, query)
        else:
            query_with_prompt_template = get_prompt("", context, query)
        # st.write("query_with_prompt_template - ", query_with_prompt_template)

        # Executing LLMs
        if model == "Gemini-1.5-flash":
            answer = llm.generate_content(query_with_prompt_template[1].blocks[0].text)
            assisstant_answers = answer.text
        else:
            answer = llm.chat(query_with_prompt_template)
            assisstant_answers = answer.message.blocks[0].text

        st.write("*"*50)
        st.write("You: ", query)
        st.write("Answers:", assisstant_answers)
        print("*"*50,"Question: ", "*"*50,"\n", )
        print(assisstant_answers)
        
        #appending query and result to the session
        history[email][chat_index].append(user_query:=query)
        history[email][chat_index].append(assisstant_answers:=assisstant_answers)


        #combined context at zeroth index of history
        history[email][chat_index][0] = "\n".join([history[email][chat_index][0], history[email][chat_index][-2], history[email][chat_index][-1]])
        print("History Updated")

        print("="*50, "end of context", "="*50)
        #feeding all the updated history into the database
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)

    
#Display Dashboard after login
def dash_page(email):
    #chat index to track the history of particular session from history of the user
    global chat_index

    #file reading
    with open(HISTORY_FILE, "r") as f:      
        history = json.load(f)

    #checking previous sessions in history
    if len(history[email]) == 0:                             
        history[email].append([""])
        chat_index = 0
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)  

    # Passing email and history to implement rag
    dash_page_rag(email, history)        
        

if __name__ == "__main__":
    dash_page(email = "sai@gmail.com")