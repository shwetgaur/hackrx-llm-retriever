# HackRx 6.0: Intelligent Query-Retrieval System

This project is an advanced Retrieval-Augmented Generation (RAG) system built for the Bajaj Finserv HackRx 6.0 hackathon. It answers natural language queries about multiple, complex insurance policy documents, providing decisions and justifications backed by specific clauses from the source texts.

## Key Features
- **Multi-Document Ingestion**: Handles multiple, large unstructured documents (PDFs) simultaneously.
- **Advanced RAG Pipeline**: Implements a sophisticated retrieval pipeline featuring:
  - **Hybrid Search**: Combines semantic (vector) search with keyword (BM25) search for robust retrieval.
  - **Re-ranking**: Uses a Cross-Encoder model to re-rank initial results for maximum relevance.
- **Reasoning Agent**: Utilizes a ReAct (Reasoning and Acting) agent that can perform multi-step logic to gather evidence before making a final decision.
- **Source-Cited Justifications**: Provides exact quotes for every decision and cites which source document the information came from.
- **API Server**: The entire system is exposed via a robust FastAPI endpoint for easy integration.

## Prerequisites
Before you begin, ensure you have the following installed on your system:
- Python (version 3.10+ recommended)
- Git

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the Repository**
```bash
git clone <your_repository_url.git>
cd <repository-name>
```

**2. Create and Activate Virtual Environment**
It is highly recommended to use a virtual environment to manage project dependencies.

* **On Windows (PowerShell):**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
* **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
Your terminal prompt should now be prefixed with `(venv)`.

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
This will install all the necessary libraries. This step can take a while, especially the first time, as it downloads large machine learning models like PyTorch.

**4. Set Up Your API Key**
The application uses the Google Gemini API for its reasoning capabilities.

* Create a new file in the root of the project named `.env`.
* Add your Google API key to this file in the following format:
    ```
    GOOGLE_API_KEY=YourSecretApiKeyGoesHere
    ```
* You can get a free API key from [Google AI Studio](https://aistudio.google.com/).

**5. Add Source Documents**
Place all the provided insurance policy PDF files into the `docs/` folder.

**6. Log in to Hugging Face (First time only)**
The project needs to download models from Hugging Face. To prevent network errors, log in using the command line.

* Get a "read" access token from [Hugging Face Tokens](https://huggingface.co/settings/tokens).
* Run the login command in your terminal:
    ```bash
    huggingface-cli login
    ```
* Paste your token when prompted.

**7. Build the Vector Database**
This script processes all documents in the `docs/` folder and creates a local vector index for fast semantic search.
```bash
python create_database.py
```
This will create a `faiss_index/` folder in your project directory.

## Running the Application

Once the setup is complete, you can start the API server.

```bash
uvicorn main:app --reload
```
The server will start and be available at `http://127.0.0.1:8000`.

*Note: You can safely ignore the `LangSmithMissingAPIKeyWarning` that appears in the terminal. It's for an optional logging service and won't affect the application's performance.*

## How to Test

The easiest way to test the API is using the auto-generated documentation.

1.  Open your web browser and navigate to:
    **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

2.  Click on the **POST `/query`** endpoint to expand it.

3.  Click the **"Try it out"** button.

4.  In the "Request body" text area, enter your query in JSON format. For example:
    ```json
    {
      "text": "46M, knee surgery, Pune, 3-month policy"
    }
    ```

5.  Click the **"Execute"** button to send the request. The JSON response from the agent will appear below.