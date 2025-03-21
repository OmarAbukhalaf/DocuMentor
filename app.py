import os
import tempfile
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pdf2image import convert_from_path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

@st.cache_resource
def load_clip_model():
    st.info("Loading CLIP model for image processing (this may take a moment)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    st.success("CLIP model loaded!")
    return model, processor

def process_document(uploaded_file: UploadedFile) -> tuple[list[Document], list[bytes]]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    text_splits = text_splitter.split_documents(docs)

    image_bytes_list = []
    images = convert_from_path(temp_file.name)
    temp_image_paths = []
    for i, img in enumerate(images):
        img_path = f"{temp_file.name}_img_{i}.png"
        img.save(img_path, "PNG")
        with open(img_path, "rb") as f:
            image_bytes_list.append(f.read())
        temp_image_paths.append(img_path)

    os.unlink(temp_file.name)
    return text_splits, image_bytes_list, temp_image_paths

def embed_images(image_paths: list[str], clip_model, clip_processor) -> list[list[float]]:
    embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        outputs = clip_model.get_image_features(**inputs)
        embeddings.append(outputs[0].detach().numpy().tolist())
    return embeddings

def get_vector_collections() -> tuple[chromadb.Collection, chromadb.Collection]:
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    text_collection = chroma_client.get_or_create_collection(
        name="rag_app_text",
        embedding_function=OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest",
        ),
        metadata={"hnsw:space": "cosine"},
    )
    image_collection = chroma_client.get_or_create_collection(
        name="rag_app_images",
        metadata={"hnsw:space": "cosine"},
    )
    return text_collection, image_collection

def add_to_vector_collection(text_splits: list[Document], image_paths: list[str], image_bytes_list: list[bytes], file_name: str, clip_model, clip_processor):
    text_collection, image_collection = get_vector_collections()

    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(text_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_text_{idx}")
    text_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    image_embeddings = embed_images(image_paths, clip_model, clip_processor)
    image_ids = [f"{file_name}_img_{i}" for i in range(len(image_paths))]
    image_metadatas = [{"index": i} for i in range(len(image_bytes_list))]  # Store index to map back to bytes
    image_collection.upsert(embeddings=image_embeddings, metadatas=image_metadatas, ids=image_ids)

    for img_path in image_paths:
        os.unlink(img_path)

    st.success("Text and images added to the vector store!")
    return image_bytes_list

def query_collection(prompt: str, clip_model, clip_processor, image_bytes_list: list[bytes], n_results: int = 10):
    text_collection, image_collection = get_vector_collections()

    text_results = text_collection.query(query_texts=[prompt], n_results=n_results)
    text_docs = text_results.get("documents")[0]

    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    query_embedding = clip_model.get_text_features(**inputs).detach().numpy().tolist()[0]
    image_results = image_collection.query(query_embeddings=[query_embedding], n_results=3)
    image_indices = [meta["index"] for meta in image_results["metadatas"][0]]
  
    relevant_image_bytes = [image_bytes_list[idx] for idx in image_indices]

    return {"text": text_docs, "images": relevant_image_bytes}

def call_llm(context: dict, prompt: str):
    text_context = " ".join(context["text"])
    image_context = " ".join([f"Image {i+1}: A visual from the document." for i in range(len(context["images"]))])
    full_context = f"Text: {text_context}\nImages: {image_context}"

    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {full_context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    return relevant_text, relevant_text_ids

if __name__ == "__main__":
    st.set_page_config(page_title="DocuMentor - Multi-Modal RAG")
    
    clip_model, clip_processor = None, None
    image_bytes_list = st.session_state.get("image_bytes_list", [])

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "**📑 Upload Lecture Notes (PDF)**", type=["pdf"], accept_multiple_files=False
        )
        process = st.button("⚡️ Process")
        if uploaded_file and process:
            if clip_model is None:
                clip_model, clip_processor = load_clip_model()
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            text_splits, image_bytes_list, temp_image_paths = process_document(uploaded_file)
            image_bytes_list = add_to_vector_collection(text_splits, temp_image_paths, image_bytes_list, normalize_uploaded_file_name, clip_model, clip_processor)
            st.session_state.image_bytes_list = image_bytes_list

    st.header("🗣️ DocuMentor - Ask About Your Notes")
    prompt = st.text_area("**Ask a question about your lecture notes:**")
    ask = st.button("🔥 Ask")

if ask and prompt:
    if clip_model is None:
        clip_model, clip_processor = load_clip_model()
    if not image_bytes_list:
        st.warning("Please process a PDF first!")
    else:
        results = query_collection(prompt, clip_model, clip_processor, image_bytes_list)
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, results["text"])
        context = {"text": [relevant_text], "images": results["images"]}

        # Check if the prompt is about displaying an image
        prompt_lower = prompt.lower()
        if "display" in prompt_lower and "image" in prompt_lower:
            st.write("Here are the relevant images from your PDF:")
            for i, img_bytes in enumerate(results["images"]):
                st.image(img_bytes, caption=f"Image {i+1}", use_container_width=True)
            if not results["images"]:
                st.write("No relevant images found for your query.")
        else:
            # Proceed with AI response for non-image-display queries
            response = call_llm(context=context, prompt=prompt)
            st.write_stream(response)

            with st.expander("See Retrieved Content"):
                st.subheader("Text")
                st.write(relevant_text)
                st.subheader("Images")
                for img_bytes in results["images"]:
                    st.image(img_bytes, caption="Relevant Image", use_container_width=True)

            with st.expander("See Most Relevant Text IDs"):
                st.write(relevant_text_ids)
