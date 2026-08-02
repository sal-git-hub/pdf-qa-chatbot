from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def load_documents(pdf_paths):
    documents = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks):
    embeddings = create_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def load_llm():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        device=-1
    )

    return HuggingFacePipeline(
        pipeline=generator
    )
