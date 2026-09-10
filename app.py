import os
import io
import pandas as pd
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import chromadb
from gtts import gTTS
import joblib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

DB_DIR = os.path.abspath("vectorstore")

st.set_page_config(page_title="EV Battery Factory Copilot", page_icon="⚡", layout="wide")

st.title("⚡ EV Battery Factory Operations Copilot")
st.caption("AI-Powered Assistant & Predictive Diagnostics for Battery Assembly Lines")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    client = chromadb.PersistentClient(path=DB_DIR)
    return Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=get_embeddings()
    )

@st.cache_resource
def get_chain(key):
    llm = ChatGroq(
        groq_api_key=key,
        model="openai/gpt-oss-20b",
        temperature=0.0,
        max_tokens=300
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Factory Operations Copilot for EV Battery Manufacturing.\n"
                   "Answer the question directly using ONLY the technical context provided below.\n"
                   "State the exact action, citing section numbers and parameter limits if given.\n"
                   "If not present in the context, respond: 'Information not available in current factory SOP documentation.'"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    return prompt | llm | StrOutputParser()

@st.cache_resource
def load_ml_model():
    if os.path.exists("health_model.joblib"):
        return joblib.load("health_model.joblib")
    return None

ml_model = load_ml_model()

if not api_key:
    st.error("GROQ_API_KEY is not configured. Please add it to your environment or Streamlit Secrets.")
    st.stop()

chain = get_chain(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# Sidebar Controls
st.sidebar.title("Factory Line Quick Actions")
st.sidebar.caption("Trigger common assembly line queries:")

selected_query = None

if st.sidebar.button("Fault Code E-101 (Humidity)"):
    selected_query = "What action is required if fault code E-101 occurs?"

if st.sidebar.button("Fault Code E-204 (Horn Misalignment)"):
    selected_query = "What action is required if fault code E-204 occurs?"

if st.sidebar.button("Fault Code E-305 (Casing Bulge)"):
    selected_query = "What action is required if fault code E-305 occurs?"

if st.sidebar.button("Fault Code E-901 (Helium Leak)"):
    selected_query = "What action is required if fault code E-901 occurs?"

if st.sidebar.button("PPE & Dry Room Requirements"):
    selected_query = "What are the required PPE and dry room environmental conditions?"

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.audit_log = []
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Shift Audit & Incident Export")

if st.session_state.audit_log:
    df_log = pd.DataFrame(st.session_state.audit_log)
    csv_buffer = io.StringIO()
    df_log.to_csv(csv_buffer, index=False)
    st.sidebar.download_button(
        label="Download Shift Log (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"factory_copilot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.sidebar.caption("No operations logged yet for export.")

st.sidebar.divider()
st.sidebar.subheader("Upload New SOP Document")
uploaded_file = st.sidebar.file_uploader("Upload .txt or .md SOP", type=["txt", "md"])

if uploaded_file is not None:
    if st.sidebar.button("Process & Ingest Document"):
        raw_text = uploaded_file.read().decode("utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.create_documents([raw_text])
        vdb = get_vector_store()
        vdb.add_documents(chunks)
        st.sidebar.success(f"Ingested {len(chunks)} chunks successfully!")
        st.rerun()

# Layout Tabs
tab_copilot, tab_ml = st.tabs(["AI Operations Copilot (RAG)", "Machine Health Diagnostics (ML)"])

# TAB 1: RAG COPILOT
with tab_copilot:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "audio" in msg and msg["audio"]:
                st.audio(msg["audio"], format="audio/mp3")
            if "sources" in msg and msg["sources"]:
                with st.expander("View Verified SOP Reference Chunks"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"Reference Chunk {idx}")
                        st.info(src)

    user_query = st.chat_input("Enter fault code or SOP query...") or selected_query

    if user_query:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        vdb = get_vector_store()
        retriever = vdb.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(user_query)
        source_texts = [d.page_content for d in docs]
        combined_context = "\n\n".join(source_texts)

        answer = chain.invoke({"context": combined_context, "question": user_query})

        audio_fp = io.BytesIO()
        tts = gTTS(text=answer, lang="en")
        tts.write_to_fp(audio_fp)
        audio_bytes = audio_fp.getvalue()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_texts,
            "audio": audio_bytes
        })
        
        st.session_state.audit_log.append({
            "Timestamp": timestamp,
            "Query": user_query,
            "Response": answer,
            "Retrieved_Sources": " | ".join(source_texts)
        })

        st.rerun()

# TAB 2: ML HEALTH PREDICTION
with tab_ml:
    st.subheader("Welding and Cell Formation Line Diagnostics")
    st.caption("Live edge telemetry feed for ultrasonic welders and formation cycling bays.")

    if ml_model is None:
        st.warning("Model file health_model.joblib not found. Run train_health_model.py first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            temp_input = st.slider("Horn / Bay Temperature (deg C)", 15.0, 75.0, 26.0, 0.5)
            vib_input = st.slider("Tool Vibration (mm/s RMS)", 0.1, 8.0, 1.2, 0.1)

        with col2:
            res_input = st.slider("Internal Resistance (mOhm)", 0.5, 8.0, 1.8, 0.1)
            volt_input = st.slider("Terminal Voltage (V)", 2.4, 4.8, 3.9, 0.05)

        sample_df = pd.DataFrame([{
            'temperature': temp_input,
            'vibration': vib_input,
            'resistance': res_input,
            'voltage': volt_input
        }])

        if st.button("Run Diagnostic Inference"):
            pred = ml_model.predict(sample_df)[0]
            probs = ml_model.predict_proba(sample_df)[0]

            labels = [
                "Normal Operational State",
                "Warning: Degradation / High Heat",
                "Critical: Immediate Failure / Anomaly"
            ]
            
            st.divider()
            if pred == 0:
                st.success(f"Status: {labels[0]}")
                st.caption(f"Normal: {probs[0]*100:.1f}% | Warning: {probs[1]*100:.1f}% | Critical: {probs[2]*100:.1f}%")
            elif pred == 1:
                st.warning(f"Status: {labels[1]}")
                st.caption(f"Normal: {probs[0]*100:.1f}% | Warning: {probs[1]*100:.1f}% | Critical: {probs[2]*100:.1f}%")
                st.info("Recommendation: Schedule routine calibration and check ultrasonic transducer cooling.")
            else:
                st.error(f"Status: {labels[2]}")
                st.caption(f"Normal: {probs[0]*100:.1f}% | Warning: {probs[1]*100:.1f}% | Critical: {probs[2]*100:.1f}%")
                st.error("Intervention Required: Temperature or vibration exceeds safe operating thresholds.")
