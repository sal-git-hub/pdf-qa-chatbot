from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.llms import HuggingFacePipeline


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
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def create_vector_store(chunks):

    embeddings = create_embeddings()

    return FAISS.from_documents(chunks, embeddings)


def load_llm():

    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)