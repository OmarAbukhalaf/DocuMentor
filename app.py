import os
import time
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from PIL import Image
import redis
import pytesseract
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'pdfs/'
figures_directory = 'multi-modal-rag/figures/'

embeddings = OllamaEmbeddings(model="llama3.2")
model = OllamaLLM(model="gemma3:4b")

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}
)

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    for file in os.listdir(figures_directory):
        extracted_text = extract_text(figures_directory + file)
        text_elements.append(extracted_text)

    return "\n\n".join(text_elements)

def extract_text(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_text(text)

def index_docs(texts):
    ids = [str(i) for i in range(len(texts))]
    embeddings_list = embeddings.embed_documents(texts)
    collection.add(ids=ids, documents=texts, embeddings=embeddings_list)
    
def retrieve_docs(query):
    cache_key = f"retrieved_docs:{query}"  

    cached_results = redis_client.get(cache_key)
    if cached_results:
        print("Cache hit! Returning cached documents.")
        return json.loads(cached_results)  

    print("Cache miss! Retrieving documents.")
    results = collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=2
    )
    retrieved_docs = [doc for doc in results['documents'][0] if doc]

    redis_client.setex(cache_key, 3600, json.dumps(retrieved_docs))

    return retrieved_docs



def answer_question(question, documents):
    cache_key = f"answer:{question}"  

    cached_answer = redis_client.get(cache_key)
    if cached_answer:
        print("Cache hit! Returning cached answer.")
        return cached_answer 

    print("Cache miss! Generating new answer.")
    context = "\n\n".join(documents)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    answer = chain.invoke({"question": question, "context": context})

    redis_client.setex(cache_key, 3600, answer)

    return answer



st.set_page_config(page_title="Chat with PDF", layout="wide")


st.sidebar.title("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    upload_pdf(uploaded_file) 
    text = load_pdf(pdfs_directory + uploaded_file.name)  
    chunked_texts = split_text(text)  
    index_docs(chunked_texts)  
    st.sidebar.success("✅ PDF uploaded and processed!")

st.title("🤖 DocuMentor")
st.write("Ask questions based on your uploaded documents.")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

question = st.chat_input("Ask a question...")
start=time.time()
if question:
    with st.chat_message("user"):
        st.write(question)  

    related_documents = retrieve_docs(question)
    print("Retrieved docs:", related_documents)
    answer = answer_question(question, related_documents)
    print(f"Time: {time.time() - start} seconds")
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
