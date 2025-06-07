# IntliQuiz: Offline PDF Chatbot
# This script allows users to upload multiple PDF files, extract text from them,
# chunk the text, create a vector store for efficient searching, and interact with the content
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from googletrans import Translator

# 1. Read text from multiple PDFs

def extract_text_from_pdfs(files):
    all_text = ""
    page_map = []  # To store (filename, page number, snippet)
    for file in files:
        file.seek(0)
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for i, page in enumerate(pdf):
            text = page.get_text()
            all_text += text
            page_map.append((file.name, i + 1, text))
    return all_text, page_map

# 2. Chunk text for vector embedding
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# 3. Create vector store from chunks
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# 4. Setup chat chain
def setup_qa_chain(vectorstore):
    llm = Ollama(model="mistral")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# 5. Translate text using googletrans
def translate(text, target_lang):
    translator = Translator()
    return translator.translate(text, dest=target_lang).text

# 6. Generate quiz from document content
def generate_quiz(content):
    prompt = (
        "Create 5 quiz questions based only on the following content.\n"
        f"CONTENT:\n{content[:1500]}\n"
        "Return only the questions."
    )
    llm = Ollama(model="mistral")
    return llm.invoke(prompt)

# 7. Keyword search across all PDFs
def search_keywords(page_map, keyword):
    matches = []
    for filename, page_num, text in page_map:
        if keyword.lower() in text.lower():
            snippet = text[text.lower().find(keyword.lower()):][:200].replace('\n', ' ')
            matches.append(f"📄 {filename} — Page {page_num}: {snippet}...")
    return matches

# --- Streamlit UI ---
st.set_page_config(page_title="📄 IntliQuiz: Offline PDF Chatbot", layout="centered")
st.title("📄 IntliQuiz — Offline PDF Chat, Quiz, Translate & Search")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
lang = st.selectbox("🌐 Translate answers to: (optional)", ["none", "hindi", "telugu", "tamil", "france", "espanol", "de", "zh"])

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        full_text, page_map = extract_text_from_pdfs(uploaded_files)
        chunks = chunk_text(full_text)
        vectorstore = create_vector_store(chunks)
        qa_chain = setup_qa_chain(vectorstore)
        st.success("PDFs processed! You can now chat, generate quizzes, or search keywords.")

        search_query = st.text_input("🔍 Enter a keyword to search across all PDFs (optional):")
        if search_query:
            matches = search_keywords(page_map, search_query)
            if matches:
                st.subheader("🔎 Search Results:")
                for match in matches:
                    st.write(match)
            else:
                st.info("No matches found.")

        question = st.text_input("💬 Ask something or type 'generate quiz'")
        if question:
            if "quiz" in question.lower():
                result = generate_quiz(full_text)
            else:
                result = qa_chain.run(question)

            if lang != "none":
                result = translate(result, lang)
            st.markdown(f"**Response:**\n{result}")
else:
    st.info("Upload one or more PDFs to begin.")