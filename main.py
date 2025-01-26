import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import logging
import re
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.error(f"Traceback: {traceback.format_exc()}")

# Define ingestion function
def ingest(file):
    try:
        logging.info("Starting PDF ingestion.")
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()
        logging.info(f"Loaded {len(pages)} pages from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(pages)
        logging.info(f"Split pages into {len(chunks)} chunks.")

        logging.info("Load FastEmbedEmbeddings")
        embedding = FastEmbedEmbeddings()

        logging.info("Dump data in Chroma db STATUS: Start")
        try:
            Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./sql_chroma_db")
            logging.info("Successfully persisted data to Chroma.")
        except Exception as e:
            logging.error(f"Error persisting data to Chroma: {e}")
        logging.info("Dump data in Chroma db STATUS: Done")
        logging.info("Saved chunks to vector store.")

        return f"Processed {len(pages)} pages into {len(chunks)} chunks and saved to vector store."
    except Exception as e:
        logging.error(f"Error during ingestion: {e}")
        return f"An error occurred during ingestion: {e}"

# Define RAG chain function
def rag_chain():
    try:
        logging.info("Creating RAG chain.")
        model = ChatOllama(model="llama3.2-vision:11b")
        prompt = PromptTemplate.from_template(
            """
            <s> [Instructions] You are a helpful and precise assistant. Use the provided context to answer the question clearly and accurately. 
            If the context does not contain enough information, respond with: "No context available for this question: {input}".
            Ensure your answer is well-structured, concise, and directly addresses the question. [/Instructions] </s>
            [Question] {input} [/Question]
            [Context] {context} [/Context]
            [Answer]
            """
        )
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 40, "score_threshold": 0.15},
        )
        document_chain = create_stuff_documents_chain(model, prompt)
        logging.info("RAG chain successfully created.")
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        logging.error(f"Error during RAG chain creation: {e}")
        raise

# Define query function with monitoring and guardrails
def ask(query):
    try:
        # Basic guardrails
        def anonymize_query(query):
            # Simple anonymization for emails
            query = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', '[REDACTED]', query)
            return query

        def contains_restricted_language(query):
            restricted_substrings = ["badword", "offensive"]
            return any(substring in query.lower() for substring in restricted_substrings)

        def contains_code_injection(query):
            return bool(re.search(r'(SELECT|INSERT|DROP|DELETE|UPDATE|--|\;)', query, re.IGNORECASE))

        # Apply guardrails
        query = anonymize_query(query)
        if contains_restricted_language(query):
            raise ValueError("Query contains restricted language.")
        if contains_code_injection(query):
            raise ValueError("Potential code injection detected.")

        logging.info(f"Received query: {query}")
        chain = rag_chain()
        if not chain:
            raise RuntimeError("RAG chain creation failed.")

        result = chain.invoke({"input": query})

        answer = result.get("answer", "No answer available.")
        context = result.get("context", [])

        if not context:
            logging.warning("No sources found in the context.")
            sources = ["No sources available."]
        else:
            sources = [doc.metadata.get("source", "Unknown source") for doc in context]

        logging.info("Query successfully processed.")
        return answer, sources
    except Exception as e:
        logging.error(f"Error in ask function: {e}")
        return "An error occurred while processing your query.", ["No sources available."]

# Gradio UI setup
def query_interface(query):
    try:
        logging.info("Processing user query through interface.")
        answer, sources = ask(query)
        return answer, sources
    except Exception as e:
        logging.error(f"Error in query interface: {e}")
        return "An error occurred while processing your query.", ["No sources available."]

def main():
    try:
        logging.info("Launching Gradio UI.")
        with gr.Blocks() as ui:
            with gr.Tab("Ingest PDF"):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                ingest_button = gr.Button("Ingest")
                ingest_output = gr.Textbox(label="Ingestion Status")

                ingest_button.click(ingest, inputs=pdf_input, outputs=ingest_output)

            with gr.Tab("Ask Question"):
                query_input = gr.Textbox(label="Enter your question:")
                query_button = gr.Button("Ask")
                query_output = gr.Markdown(label="Answer")
                source_output = gr.Textbox(label="Sources")

                query_button.click(query_interface, inputs=query_input, outputs=[query_output, source_output])

        ui.launch(share=True)
    except Exception as e:
        logging.error(f"Error during UI launch: {e}")

if __name__ == "__main__":
    main()
