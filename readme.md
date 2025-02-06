# RAG for Requirement Document Analysis

## **Objective**

To create a Retrieval-Augmented Generation (RAG)-based tool that helps IT service providers efficiently analyze and interpret client-provided requirement documents (detailed PDFs). This tool is designed to:

- Extract key requirements with unmatched precision and clarity.
- Provide a detailed and comprehensive summary of the client’s specific needs and objectives.
- Answer complex and detailed queries about the document with a high level of accuracy.
- Offer actionable solutions or suggest follow-up questions to address ambiguities or incomplete information within the document.
- Streamline collaboration between teams by enhancing the overall requirement management process.

---

## **Setup and Installation**

Before implementing the RAG system, it is crucial to set up the environment properly. This section outlines the necessary steps to install, configure, and prepare all required components, ensuring the system runs efficiently on either CPU or GPU.

### **1. Setting up Ollama with Docker**

Ollama is used to host the Llama3.2-Vision:11b model, enabling natural language processing capabilities essential for this RAG tool. Below are the steps to set up Ollama with Docker:

#### **Step 1: Install Docker**

- Visit the [Docker official website](https://www.docker.com/) and download the appropriate version for your operating system.
- Follow the installation instructions specific to your platform (Windows, macOS, or Linux).
- After installation, verify Docker by running the following command in your terminal or command prompt:
  ```bash
  docker --version
  ```

#### **Step 2: Launch Docker and Pull the Ollama Image**

1. Open your terminal or command prompt.
2. Run the following command to pull the Ollama Docker image:
   ```bash
   docker pull ollama/ollama:latest
   ```
3. Start a Docker container for Ollama:
   - If you have GPU support available and wish to utilize it for optimal performance, use:
     ```bash
     docker run --gpus all -d --name ollama-container -p 11434:11434 ollama/ollama:latest
     ```
     - `--gpus all`: Enables GPU usage for accelerated performance.
     - `-p 11434:11434`: Maps the container's port 11434 to your host machine.
     - Ensure that the CUDA Toolkit is installed on your system for GPU functionality. For installation, refer to [CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit).

   - If you prefer to use CPU, omit the `--gpus all` flag:
     ```bash
     docker run -d --name ollama-container -p 11434:11434 ollama/ollama:latest
     ```

#### **Step 3: Verify the Docker Container**

- Check if the Ollama container is running by executing:
  ```bash
  docker ps
  ```
  You should see `ollama-container` listed in the output.

### **2. Downloading the Llama3.2-Vision:11b Model**

The Llama3.2-Vision:11b model must be downloaded within the Docker container to enable advanced natural language processing. Follow these steps:

1. Access the running Ollama container:
   ```bash
   docker exec -it ollama-container bash
   ```
2. Use the Ollama CLI to download the required model:
   ```bash
   ollama pull llama3.2-vision:11b
   ```
   This command fetches the Llama3.2-Vision:11b model and prepares it for use.
3. Exit the container once the model is successfully downloaded:
   ```bash
   exit
   ```
![Docker Screenshot](/assets/docker1.webp "")

### **3. Installing Required Python Libraries**

Python libraries play a crucial role in the implementation of this RAG system. Install them as follows:

1. Ensure Python 3.8 or higher is installed on your system. Verify your version by running:
   ```bash
   python --version
   ```
2. (Optional but recommended) Create a virtual environment to isolate dependencies:
   ```bash
   python -m venv rag-env
   source rag-env/bin/activate    # For Linux/Mac
   .\rag-env\Scripts\activate  # For Windows
   ```
3. Install the necessary libraries:
   ```bash
   pip install gradio langchain langchain-chroma langchain-community fastembed langchain-ollama
   ```
4. Verify that all libraries have been installed correctly:
   ```bash
   pip list
   ```

---

## **Implementation Overview**

### Core Libraries and Tools Used

- **Gradio**: Provides a dynamic, user-friendly interactive interface for document ingestion and querying.
- **LangChain**: Powers the chaining of document ingestion, embedding, and retrieval processes to enable seamless execution.
- **Chroma**: Acts as the vector database for storing and managing embeddings.
- **PyPDFLoader**: Handles the extraction and splitting of content from multi-page PDF files into smaller, manageable chunks.
- **ChatOllama**: Integrates the Llama3.2-Vision:11b model, offering advanced natural language processing capabilities.
- **FastEmbedEmbeddings**: Generates high-quality embeddings for document indexing and retrieval.

---

## **Code Implementation**

### **PDF Ingestion and Vector Storage**

**Description**: This process ingests a PDF document, splits it into smaller chunks for better processing, and saves the processed data into a vector database (Chroma). This ensures efficient and accurate retrieval of relevant information.

```python
import logging
import traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.error(f"Traceback: {traceback.format_exc()}")

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

        logging.info("Loading FastEmbedEmbeddings.")
        embedding = FastEmbedEmbeddings()

        logging.info("Dumping data into Chroma database.")
        try:
            Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./sql_chroma_db")
            logging.info("Successfully persisted data to Chroma.")
        except Exception as e:
            logging.error(f"Error persisting data to Chroma: {e}")

        return f"Processed {len(pages)} pages into {len(chunks)} chunks and saved to vector store."
    except Exception as e:
        logging.error(f"Error during ingestion: {e}")
        return f"An error occurred during ingestion: {e}"
```

![Ingest Screenshot](/assets/SS-ingest.webp "")

---

### **RAG Chain Creation**

**Description**: Configures the Retrieval-Augmented Generation (RAG) chain using the Llama3.2-Vision:11b model to enable natural language querying and precise response generation.

```python
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


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
            search_kwargs={"k": 20, "score_threshold": 0.15},
        )
        document_chain = create_stuff_documents_chain(model, prompt)
        logging.info("RAG chain successfully created.")
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        logging.error(f"Error during RAG chain creation: {e}")
        raise
```

---

### **Query Processing**

**Description**: Processes user queries, retrieves relevant information from the vector database, and generates accurate responses using the RAG chain.

```python
def ask(query):
    try:
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
```

![Chat Screenshot](/assets/SS-chat.webp "")

---

### **Gradio UI**

**Description**: Provides an intuitive Gradio interface for document ingestion and querying.

```python
import gradio as gr

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
                source_output = gr.Textbox
                                source_output = gr.Textbox(label="Sources")

                query_button.click(query_interface, inputs=query_input, outputs=[query_output, source_output])

        ui.launch()
    except Exception as e:
        logging.error(f"Error during UI launch: {e}")

if __name__ == "__main__":
    main()
```
---

## Security with Guardrails

To ensure Responsible AI, security guardrails are implemented to filter bad queries and prevent misuse. Below are the applied techniques:

### 1. **Anonymization**
- **Objective**: Automatically remove personally identifiable information (PII) from user queries.

**Code Example**:
```python
from langchain_guardrails import Anonymizer

anonymizer = Anonymizer()
query = "John Doe's email is john.doe@example.com"
filtered_query = anonymizer.anonymize(query)
print(f"Filtered Query: {filtered_query}")
```

### 2. **Restrict Substrings**
- **Objective**: Prevent malicious queries containing restricted substrings like profanity or offensive terms.

**Code Example**:
```python
restricted_substrings = ["badword", "offensiveterm"]
query = "This is a badword test."
if any(substring in query for substring in restricted_substrings):
    raise ValueError("Query contains restricted content.")
```

### 3. **Restrict Language**
- **Objective**: Block inappropriate or harmful language queries.

**Code Example**:
```python
from langchain_guardrails import LanguageFilter

language_filter = LanguageFilter(allowed_languages=["en"])
query = "Este es un texto en español."
if not language_filter.is_allowed(query):
    raise ValueError("Language not allowed.")
```

### 4. **Restrict Code Injection**
- **Objective**: Prevent injection attacks through malicious code in queries.

**Code Example**:
```python
from langchain_guardrails import InjectionFilter

injection_filter = InjectionFilter()
query = "SELECT * FROM users;"
if injection_filter.contains_injection(query):
    raise ValueError("Potential code injection detected.")
```

---

## Evaluation of Responses

### Evaluation Aspects
To ensure the RAG system delivers high-quality and responsible responses, the following evaluation criteria are applied:

1. **Fairness**
   - **Goal**: Ensure that the responses do not reflect unjust biases based on gender, race, or other attributes.
   - **Evaluation Method**: Test queries are crafted to analyze how the model responds to diverse scenarios.

   **Code Example**:
   ```python
   test_queries = [
       "Provide maternity leave requirements.",
       "Describe policies for paternity leave."
   ]

   for query in test_queries:
       answer, sources = ask(query)
       print(f"Query: {query}\nAnswer: {answer}\nSources: {sources}\n")
   ```

2. **No Unjust Bias**
   - **Goal**: Avoid biases in retrieved content or generated responses.
   - **Evaluation Method**: Regularly audit the vector database for biased documents and incorporate fairness tests in the RAG chain evaluation.

3. **Relevance and Accuracy**
   - **Goal**: Ensure the response is directly aligned with the query context.
   - **Evaluation Method**: Precision and recall metrics are used for the retrieved documents, combined with manual checks for correctness.

4. **Handling Ambiguity**
   - **Goal**: Generate follow-up questions for incomplete or unclear queries.
   - **Evaluation Method**: Test queries with incomplete context to evaluate if the system provides actionable follow-up questions.

### Sample Evaluation
- **Query**: "Explain the requirements for data privacy compliance."

**Code Example**:
```python
query = "Explain the requirements for data privacy compliance."
answer, sources = ask(query)
print(f"Answer: {answer}\nSources: {sources}")
```

---

### **Applications of RAG in IT Services**

RAG systems can significantly streamline processes in the IT services industry, especially for requirement gathering, analysis, and project initiation. Some of the key applications include:

1. **Streamlined Requirement Analysis**: Quickly analyze lengthy client-provided documents and identify critical project details.
2. **Proposal Generation**: Automatically generate initial drafts of project proposals based on extracted information from client requirements.
3. **Client Communication**: Enhance communication by providing instant answers to client queries based on stored project data.
4. **Knowledge Retention**: Build a knowledge base by storing project-related documents and their processed outputs for future reference.
5. **Automated Follow-Ups**: Identify gaps in client requirements and generate follow-up questions for clarification.


### Visit [GitHub](https://github.com/Konohamaru04/RAG-for-Requirement-Document-Analysis) for codebase.
---

### **Challenges and How to Overcome Them**

Implementing a RAG system involves several challenges. Below are common issues and their solutions:

#### **1. Data Quality Issues**
- **Challenge**: Poorly formatted or inconsistent data in client-provided documents.
- **Solution**: Use advanced preprocessing tools like PyPDFLoader and text splitters to clean and standardize the data before ingestion.

#### **2. Scalability**
- **Challenge**: Managing large volumes of documents and queries efficiently.
- **Solution**: Employ a scalable vector database like Chroma and optimize storage with techniques such as chunking.

#### **3. Accuracy of Results**
- **Challenge**: Generating relevant and accurate responses.
- **Solution**: Fine-tune the Llama3.2-Vision model with domain-specific data and use retrieval parameters like "mmr" to enhance context relevance.

#### **4. Integration Complexity**
- **Challenge**: Integrating the RAG system into existing IT workflows and tools.
- **Solution**: Design modular APIs and provide a user-friendly interface like Gradio for seamless integration.

---

## Why RAG Was Chosen Over Fine-Tuning

In this use case, Retrieval-Augmented Generation (RAG) was preferred over fine-tuning for several reasons:

1. **Dynamic Updates**: RAG leverages an external vector database for retrieval, allowing the system to dynamically incorporate new information without requiring model retraining.

2. **Cost Efficiency**: Fine-tuning large models like Llama3.2-Vision:11b can be resource-intensive, requiring significant computational power and time. RAG avoids this by using pre-trained models and focusing on retrieval.

3. **Adaptability**: Fine-tuned models are often tailored to specific tasks and may underperform on out-of-scope queries. RAG, with its retrieval-based approach, is inherently more flexible and adaptable to a wider range of queries.

4. **Domain Independence**: By separating the knowledge base (vector database) from the model, RAG allows the knowledge base to be domain-specific while keeping the model generic, thus enabling better scalability and reuse.

5. **Faster Deployment**: RAG-based solutions can be deployed quickly by populating a vector store with embeddings, whereas fine-tuning requires additional cycles for data preparation, training, and evaluation.

---

### **Future Enhancements and Business Benefits**

#### **1. Multilingual Support**
- **Enhancement**: Extend the system to process and respond in multiple languages.
- **Business Benefit**: Serve global clients more effectively by breaking language barriers.

#### **2. Real-Time Collaboration**
- **Enhancement**: Enable multiple users to interact with the system simultaneously.
- **Business Benefit**: Boost team collaboration and decision-making efficiency.

#### **3. Advanced Analytics**
- **Enhancement**: Provide detailed analytics on requirement trends and gaps.
- **Business Benefit**: Help stakeholders make data-driven decisions and plan projects effectively.

---

## Next Steps for Production Deployment (On-Premises System)

### 1. **Infrastructure Setup**
   - **Hardware Specifications**:
     - **Servers**: 2-3 dedicated physical servers with dual Intel Xeon or AMD EPYC processors, 128GB RAM, and multiple NVIDIA A100 GPUs (if GPU acceleration is required).
     - **Storage**: RAID-10 configured SSDs (at least 4TB capacity) for data reliability and fast I/O.
     - **Networking**: 10Gbps network cards for fast intra-network communication.
   - **Network Setup**:
     - Create a VLAN specifically for the RAG system to isolate its traffic.
     - Use a load balancer (e.g., HAProxy or Nginx) to distribute traffic across multiple servers.

### 2. **Data Storage**
   - Use **PostgreSQL** or **MySQL** for metadata storage and Chroma for vector storage.
   - **Encryption**:
     - Apply AES-256 encryption for data at rest.
     - Use SSL/TLS for data in transit.
   - Set up automated database snapshots and backup to a secure network file system (NFS).

### 3. **Model Hosting**
   - Install **NVIDIA CUDA Toolkit** (v11.x or higher) for GPU acceleration.
   - Use **Docker** to containerize the Llama3.2-Vision model and its dependencies.
   - Example Docker command:
     ```bash
     docker run --gpus all -d --name llama-model -p 8000:8000 llama3.2-vision:11b
     ```

### 4. **Access Control**
   - Set up **LDAP** or **Active Directory (AD)** for centralized user management.
   - Implement **role-based access control (RBAC)**:
     - Example roles: Admin, Data Ingestor, Query User.
   - Log all user activities for auditing purposes.

### 5. **CI/CD Pipeline**
   - Use **GitLab CI/CD** or **Jenkins**:
     - Automate code builds and deployments.
     - Test model updates in a staging environment before moving to production.
     - Automate container image builds for deployment consistency.

### 6. **Monitoring and Observability**
   - Install **Prometheus** for metric collection and **Grafana** for visualization.
   - Metrics to monitor:
     - Query latency
     - GPU/CPU utilization
     - Disk I/O performance
     - Error rates
   - Set up alerts using **Alertmanager** for system failures.

### 7. **Security and Compliance**
   - Apply **firewall rules** to restrict external access.
   - Regularly run vulnerability scans using tools like **Nessus**.
   - Enforce compliance with regulations (e.g., GDPR, HIPAA) by:
     - Logging user queries for auditing.
     - Implementing data retention policies.

### 8. **Backup and Disaster Recovery**
   - Use **rsync** or backup tools to store encrypted snapshots offsite.
   - Perform regular disaster recovery drills to ensure backups are recoverable.

### 9. **Scaling and Optimization**
   - Add more GPUs or nodes if query latency exceeds acceptable thresholds.
   - Use caching mechanisms for frequently queried documents or embeddings.

### 10. **User Training and Documentation**
   - Develop a detailed user manual covering:
     - Query best practices
     - Error handling
   - Conduct quarterly training sessions for end-users and IT staff.
   - Maintain an internal **FAQ repository** to address common issues.

---

### **Conclusion**

The integration of Retrieval-Augmented Generation (RAG) into IT services has the potential to transform how client requirements are processed and managed. By leveraging tools like LangChain, Chroma, and Llama3.2-Vision, IT service providers can offer faster, more accurate, and scalable solutions to their clients. With continuous enhancements, RAG systems can serve as a cornerstone for intelligent project management and client engagement.

---




