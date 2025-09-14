import streamlit as st
import os
import re
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
import whisper
import tempfile
import time
import json
import requests
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import firebase_admin
from firebase_admin import credentials, auth, firestore
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES FROM .env FILE ---
load_dotenv()

# --- FIREBASE SETUP & SECURE AUTHENTICATION ---
def init_firebase():
    """Initializes the Firebase Admin SDK."""
    try:
        if not firebase_admin._apps:
            # The filename is now loaded from the .env file
            cred = credentials.Certificate(os.environ.get("FIREBASE_KEY_FILENAME"))
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase initialization failed. Make sure your key file is in the project folder and the filename in your .env file is correct. Error: {e}")
        return None

def verify_password(email, password):
    """Securely verifies user password using Firebase Auth REST API."""
    api_key = os.environ.get("FIREBASE_WEB_API_KEY")
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(rest_api_url, json=payload)
    if response.status_code == 200: 
        return response.json()
    else:
        raise Exception(response.json().get("error", {}).get("message", "Login failed."))

# --- FUTURISTIC UI STYLES & ANIMATIONS ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@400;700&display=swap');
        html, body, [class*="st-"] { font-family: 'Roboto Mono', monospace; color: #E0E0E0; }
        .stApp { background-image: linear-gradient(to bottom right, #020418, #0A0A0A); }
        h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #FFFFFF; }
        .main-title { font-size: 3.5rem; text-align: center; animation: glow 2s ease-in-out infinite alternate, flicker 5s linear infinite; }
        @keyframes glow {
            from { text-shadow: 0 0 10px #0077be, 0 0 20px #0077be, 0 0 30px #0077be; }
            to { text-shadow: 0 0 20px #00a8e8, 0 0 30px #00a8e8, 0 0 40px #00a8e8; }
        }
        @keyframes flicker {
            0%, 18%, 22%, 25%, 53%, 57%, 100% { text-shadow: 0 0 10px #0077be, 0 0 20px #0077be, 0 0 30px #0077be, 0 0 40px #00a8e8, 0 0 70px #00a8e8, 0 0 80px #00a8e8, 0 0 100px #00a8e8; }
            20%, 24%, 55% { text-shadow: none; }
        }
        .stButton>button {
            background-image: linear-gradient(to right, #00A8E8 0%, #007EA7 51%, #00A8E8 100%);
            color: #FFFFFF; border: none; border-radius: 10px; padding: 15px 30px; font-weight: bold;
            font-family: 'Orbitron', sans-serif; transition: 0.5s; background-size: 200% auto;
            box-shadow: 0 0 20px rgba(0, 168, 232, 0.4);
        }
        .stButton>button:hover { background-position: right center; box-shadow: 0 0 30px rgba(0, 168, 232, 0.7); transform: scale(1.05); }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { background-color: rgba(30, 30, 30, 0.8); color: #E0E0E0; border: 1px solid #00A8E8; border-radius: 8px; }
        .stFileUploader>div>div>button { background-color: rgba(30, 30, 30, 0.8); color: #E0E0E0; border: 1px dashed #00A8E8; }
        .st-emotion-cache-1f1G2gn { background-color: #003459; border-radius: 15px 15px 0 15px; border: 1px solid #007EA7; }
        .st-emotion-cache-4oy321 { background-color: #1E1E1E; border-radius: 15px 15px 15px 0; border: 1px solid #333; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .stChatMessage { animation: fadeIn 0.5s ease-in-out; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA INGESTION ENGINE ---
def get_pdf_data(pdf_docs):
    all_text, all_tables = "", []
    for pdf in pdf_docs:
        pdf.seek(0)
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:
                all_text += page.get_text() + "\n"
                for table in page.find_tables():
                    try:
                        if not (df := table.to_pandas()).empty: all_tables.append(df)
                    except: continue
    return all_text, all_tables
def get_text_from_txt(txt_docs):
    all_text = ""
    for txt in txt_docs:
        txt.seek(0)
        all_text += txt.read().decode("utf-8") + "\n"
    return all_text
def get_text_from_audio(audio_files):
    st.write("🎤 Transcribing audio...")
    all_text = ""
    whisper_model = whisper.load_model("base")
    for audio_file in audio_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp:
            tmp.write(audio_file.getvalue())
            tmp_path = tmp.name
        try:
            result = whisper_model.transcribe(tmp_path, fp16=False)
            all_text += f"Transcript from {audio_file.name}:\n{result['text']}\n\n"
        finally:
            os.remove(tmp_path)
    return all_text
def get_description_from_image(image_docs):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    descriptions = ""
    for image in image_docs:
        img = Image.open(image)
        vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = "Describe this image in exhaustive detail."
        try:
            response = vision_model.generate_content([prompt, img])
            descriptions += f"Visual analysis of '{image.name}':\n{response.text}\n\n"
        except Exception as e:
            st.error(f"Could not analyze image {image.name}: {e}")
    return descriptions

# --- CORE AI & AGENT ENGINE ---
def get_text_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300).split_text(text)

# --- THE NEW SESSION-BASED ARCHITECTURE ---
# These functions are no longer cached globally. They are created per-user session.
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, convert_system_message_to_human=True)

def get_researcher_agent(vectorstore, text_chunks, llm):
    template = "You are a helpful AI assistant... Here is the context:\n{context}\nHere is the user's question:\n{question}\n\nHelpful Answer:"
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    ensemble_retriever = EnsembleRetriever(retrievers=[BM25Retriever.from_texts(text_chunks), vectorstore.as_retriever(search_kwargs={'k': 5})], weights=[0.5, 0.5])
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=ensemble_retriever, return_source_documents=True, combine_docs_chain_kwargs={"prompt": QA_PROMPT})

def get_datasci_agent(dfs, llm):
    if not dfs: return None
    main_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return create_pandas_dataframe_agent(llm, main_df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

def get_planner_chain(llm):
    template = "You are an expert research analyst... Topic: {topic}\n\nQuestions:"
    prompt = PromptTemplate(template=template, input_variables=["topic"])
    return LLMChain(prompt=prompt, llm=llm)

def get_synthesizer_chain(llm):
    template = "You are an expert report writer... Original Research Topic: {topic}\n\nCollected Research (Questions and Answers):\n{research_data}\n\nFinal Summary Report:"
    prompt = PromptTemplate(template=template, input_variables=["topic", "research_data"])
    return LLMChain(prompt=prompt, llm=llm)

def route_query(query, researcher_agent, datasci_agent):
    routing_llm = get_llm()
    prompt = f"Analyze the user's query... User Query: \"{query}\". Respond with only 'Researcher' or 'Data Scientist'."
    decision = routing_llm.invoke(prompt).content.strip()
    st.info(f"🧠 Cognitive Core: Routing to **{decision}** agent.")
    if "Data Scientist" in decision and datasci_agent:
        try:
            return {"answer": datasci_agent.invoke(query)["output"], "sources": []}
        except Exception as e:
            return {"answer": f"The Data Scientist agent encountered an error: {e}", "sources": []}
    else:
        return researcher_agent.stream({"question": query, "chat_history": []})

# --- STREAMLIT USER INTERFACE ---
def main():
    st.set_page_config(page_title="INTELLIRAG 🧠", layout="wide")
    load_css()
    st.session_state.setdefault("user", None)
    db = init_firebase()
    if not db: st.stop()

    if not st.session_state.user:
        st.markdown("<h1 class='main-title'>INTELLIRAG</h1>", unsafe_allow_html=True)
        st.subheader("Login or Create an Account")
        choice = st.selectbox("Choose Action", ["Login", "Sign Up"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if choice == "Sign Up":
            if st.button("Create Account", use_container_width=True):
                try:
                    user = auth.create_user(email=email, password=password)
                    st.success("Account created successfully! Please log in.")
                except Exception as e: st.error(f"Error creating account: {e}")
        if choice == "Login":
            if st.button("Login", use_container_width=True):
                try:
                    user_data = verify_password(email, password)
                    st.session_state.user = {"uid": user_data["localId"], "email": user_data["email"]}
                    st.success("Logged in successfully!")
                    st.rerun()
                except Exception as e: st.error(f"Login failed: {e}")
    else:
        # Initialize session state keys for a logged-in user
        st.session_state.setdefault("researcher_agent", None)
        st.session_state.setdefault("datasci_agent", None)
        st.session_state.setdefault("chat_history", [])
        st.session_state.setdefault("planner_chain", None)
        st.session_state.setdefault("synthesizer_chain", None)
        st.session_state.setdefault("research_plan", None)
        st.session_state.setdefault("view", "dashboard") # Start at the dashboard

        st.sidebar.success(f"Logged in as {st.session_state.user['email']}")
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        user_doc_ref = db.collection("users").document(st.session_state.user["uid"])
        user_doc = user_doc_ref.get()

        if st.session_state.view == "dashboard":
            st.title("Dashboard")
            st.write("Welcome to your Cognitive Core.")
            if user_doc.exists and "knowledge_base_text" in user_doc.to_dict():
                st.info("You have an existing knowledge base saved.")
                if st.button("Converse with Existing Knowledge Base"):
                    with st.spinner("Loading your previous session..."):
                        user_data = user_doc.to_dict()
                        text_chunks = get_text_chunks(user_data["knowledge_base_text"])
                        all_tables = [pd.read_json(table_json) for table_json in user_data.get("tables", [])]
                        vectorstore = get_vectorstore(text_chunks)
                        llm = get_llm()
                        st.session_state.researcher_agent = get_researcher_agent(vectorstore, text_chunks, llm)
                        st.session_state.datasci_agent = get_datasci_agent(all_tables, llm)
                        st.session_state.planner_chain = get_planner_chain(llm)
                        st.session_state.synthesizer_chain = get_synthesizer_chain(llm)
                        st.session_state.chat_history = json.loads(user_data.get("chat_history", "[]"))
                        st.session_state.view = "chat"
                        st.rerun()
            else:
                st.info("You do not have a knowledge base. Let's build one.")

            if st.button("Create New Knowledge Base"):
                with st.spinner("Executing Amnesia Protocol..."):
                    if user_doc.exists: user_doc_ref.delete()
                    keys_to_clear = ["researcher_agent", "datasci_agent", "chat_history", "planner_chain", "synthesizer_chain", "research_plan"]
                    for key in keys_to_clear:
                        if key in st.session_state: del st.session_state[key]
                st.session_state.view = "upload"
                st.rerun()

        elif st.session_state.view == "upload":
            st.subheader("Add Your Sources to Build the Knowledge Base")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📁 Upload Files")
                pdf_docs = st.file_uploader("PDFs", accept_multiple_files=True, type="pdf")
                txt_docs = st.file_uploader("Text Files", accept_multiple_files=True, type="txt")
                image_docs = st.file_uploader("Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
                audio_docs = st.file_uploader("Audio", accept_multiple_files=True, type=['mp3', 'wav', 'm4a'])
            with col2:
                st.markdown("##### ✍️ Raw Text")
                pasted_text = st.text_area("Paste Text Here", height=350)
            if st.button("Build Knowledge Base", use_container_width=True):
                with st.spinner("Processing all sources..."):
                    all_text, all_tables = "", []
                    if pdf_docs:
                        pdf_text, pdf_tables = get_pdf_data(pdf_docs)
                        all_text += pdf_text; all_tables.extend(pdf_tables)
                    if txt_docs: all_text += get_text_from_txt(txt_docs)
                    if pasted_text: all_text += pasted_text
                    if audio_docs: all_text += get_text_from_audio(audio_docs)
                    if image_docs: all_text += get_description_from_image(image_docs)
                    if all_text.strip() or all_tables:
                        text_chunks = get_text_chunks(all_text)
                        vectorstore = get_vectorstore(text_chunks)
                        llm = get_llm()
                        st.session_state.researcher_agent = get_researcher_agent(vectorstore, text_chunks, llm)
                        st.session_state.datasci_agent = get_datasci_agent(all_tables, llm)
                        st.session_state.planner_chain = get_planner_chain(llm)
                        st.session_state.synthesizer_chain = get_synthesizer_chain(llm)
                        user_doc_ref.set({
                            "knowledge_base_text": all_text,
                            "tables": [table.to_json() for table in all_tables],
                            "chat_history": json.dumps([])
                        })
                        st.session_state.chat_history = []
                        st.session_state.view = "chat"
                        st.success("Knowledge base built and saved!")
                        st.rerun()
                    else: st.warning("Could not extract any information.")

        elif st.session_state.view == "chat":
            if st.sidebar.button("Back to Dashboard"):
                st.session_state.view = "dashboard"
                st.rerun()
            tab1, tab2 = st.tabs(["💬 Chat", "📊 Autonomous Analysis"])
            with tab1:
                st.subheader("Converse with the Cognitive Core")
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message.get("sources"):
                            with st.expander("Show Sources"):
                                for doc_data in message["sources"]: st.info(doc_data["page_content"])
                if user_question := st.chat_input("Ask a question..."):
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    with st.chat_message("user"): st.markdown(user_question)
                    with st.chat_message("assistant"):
                        message_placeholder, full_response, sources = st.empty(), "", []
                        response_obj = route_query(user_question, st.session_state.researcher_agent, st.session_state.datasci_agent)
                        if isinstance(response_obj, dict):
                            full_response = response_obj["answer"]
                            sources = response_obj.get("source_documents", [])
                            message_placeholder.markdown(full_response)
                        else:
                            for chunk in response_obj:
                                if "answer" in chunk:
                                    full_response += chunk["answer"]; message_placeholder.markdown(full_response + "▌")
                                if "source_documents" in chunk:
                                    sources = chunk["source_documents"]
                            message_placeholder.markdown(full_response)

                        serializable_sources = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in sources] if sources else []
                        if serializable_sources:
                            with st.expander("Show Sources"):
                                for doc_data in serializable_sources: st.info(doc_data["page_content"])

                        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "sources": serializable_sources})
                        db.collection("users").document(st.session_state.user["uid"]).update({"chat_history": json.dumps(st.session_state.chat_history)})
            with tab2:
                st.subheader("Generate an Automated Report")
                research_topic = st.text_input("Enter a research topic", key="research_topic_input")
                if st.button("Generate Plan"):
                    if "planner_chain" in st.session_state and st.session_state.planner_chain and research_topic:
                        with st.spinner("The AI is thinking of a research plan..."):
                            plan_response = st.session_state.planner_chain.invoke({"topic": research_topic})
                            st.session_state.research_plan = plan_response['text']
                    else:
                        st.warning("Please process documents and enter a topic first.")
                if "research_plan" in st.session_state and st.session_state.research_plan:
                    st.info("AI-Generated Research Plan:")
                    st.markdown(st.session_state.research_plan)
                    if st.button("Generate Full Report from this Plan"):
                        research_data = []
                        with st.status("Executing research plan...", expanded=True) as status:
                            questions = [q.strip() for q in re.findall(r'^\d+\.\s*(.*)', st.session_state.research_plan, re.MULTILINE)]
                            for i, question in enumerate(questions):
                                st.write(f"Answering question {i+1}/{len(questions)}: *{question}*")
                                response = st.session_state.researcher_agent.invoke({"question": question, "chat_history": []})
                                research_data.append(f"Question: {question}\nAnswer: {response['answer']}")
                            status.update(label="Synthesizing final report...", state="running")
                            research_data_str = "\n\n---\n\n".join(research_data)
                            final_report = st.session_state.synthesizer_chain.invoke({"topic": st.session_state.research_topic_input, "research_data": research_data_str})
                            status.update(label="Report Generation Complete!", state="complete", expanded=False)
                        st.subheader("Final Synthesized Report")
                        st.markdown(final_report['text'])
                        st.session_state.research_plan = None

if __name__ == '__main__':
    main()