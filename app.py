import tempfile
import streamlit as st

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from rag import (
    load_documents,
    split_documents,
    create_vector_store,
    load_llm
)

st.set_page_config(
    page_title="PDF QA Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Question Answering Chatbot")

st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Choose one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success("Uploaded Files")

    for file in uploaded_files:
        st.sidebar.write(f"📄 {file.name}")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.qa = None
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa" not in st.session_state:
    st.session_state.qa = None

if uploaded_files and st.session_state.qa is None:

    pdf_paths = []

    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(uploaded_file.read())
        temp_file.close()
        pdf_paths.append(temp_file.name)

    with st.spinner("Processing PDFs... Please wait."):

        documents = load_documents(pdf_paths)

        chunks = split_documents(documents)

        vector_store = create_vector_store(chunks)

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        llm = load_llm()

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded PDFs."

Context:
{context}

Question:
{input}
"""
        )

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        st.session_state.qa = create_retrieval_chain(
            retriever,
            document_chain
        )

    st.success("PDFs processed successfully!")

if not uploaded_files:
    st.info("Upload one or more PDF files using the sidebar to begin.")

for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)

if st.session_state.qa:

    question = st.chat_input("Ask a question about your PDFs")

    if question:

        st.chat_message("user").write(question)
        st.session_state.messages.append(("user", question))

        with st.spinner("Searching documents..."):

            result = st.session_state.qa.invoke(
                {"input": question}
            )

            answer = result["answer"]

        st.chat_message("assistant").write(answer)
        st.session_state.messages.append(("assistant", answer))

        with st.expander("📚 Source Chunks"):

            for i, doc in enumerate(result["context"], start=1):

                st.markdown(f"### Source {i}")

                st.write(doc.page_content)

                st.divider()
