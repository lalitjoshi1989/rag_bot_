# FAISS Chatbot Project Documentation

## Overview
This project consists of two main components:
1. **Data Processing (Jupyter Notebook)** - Scraping and cleaning data from the web.
2. **Chatbot Application (Flask & FAISS)** - Using FAISS as a vector database to retrieve relevant information and generate responses using OpenAI's GPT model.

## Project Workflow
1. **Scraping & Cleaning Data**
   - The Jupyter Notebook scrapes data from URLs and processes it by removing unnecessary characters and formatting text.
   - The cleaned text is then split into smaller chunks for efficient retrieval.
   - Each chunk is embedded using a Sentence Transformer model and stored in a FAISS index.

2. **Chatbot Integration**
   - A Flask application loads the FAISS index and preprocessed text chunks.
   - When a user asks a question, the query is embedded and searched in the FAISS index.
   - The most relevant text chunks are retrieved and used as context for the GPT-based chatbot.

---

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- Jupyter Notebook
- Flask
- FAISS
- Sentence Transformers
- OpenAI API Key (for chatbot responses)

### Install Dependencies
```bash
pip install flask faiss-cpu sentence-transformers langchain openai numpy requests
```

### Environment Variables
Create a `.env` file to store your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

---

## Running the Project
### Step 1: Data Processing (Jupyter Notebook)
- Run the Jupyter Notebook to:
  - Scrape and clean data.
  - Convert text into embeddings using `all-MiniLM-L6-v2`.
  - Store embeddings in a FAISS index.
  - Save processed text chunks and FAISS index as files.

### Step 2: Run the Chatbot (Flask)
1. Start the Flask app:
```bash
python app.py
```
2. Open a browser and go to `http://127.0.0.1:5398`

---

## File Structure
```
faiss_chatbot_project/
│── data_processing.ipynb  # Jupyter Notebook for scraping & cleaning
│── app.py                 # Flask server
│── faiss_index.bin        # FAISS vector database
│── text_chunks.pkl        # Processed text chunks
│── templates/
│   └── index.html         # Chatbot UI
│── .env                   # API key
│── requirements.txt       # Dependencies
```

---

## Troubleshooting
### Common Issues
1. **FAISS Index Not Found**
   - Ensure `faiss_index.bin` exists before running `app.py`.
   - If missing, rerun `data_processing.ipynb` to regenerate it.

2. **Bot Replies "undefined" in UI**
   - Check that `data.response` is correctly received from the Flask API.
   - Verify Flask logs for errors (`print(response.json())` in JavaScript).

3. **OpenAI API Errors**
   - Ensure your API key is valid (`.env` file correctly set up).
   - Check OpenAI API rate limits.

---

## Future Enhancements
- Improve text preprocessing for better search results.
- Add more efficient embedding models.
- Deploy as a production-ready web service.

---

## License
This project is open-source under the MIT License.

---

## Contact
For issues or suggestions, please reach out via GitHub Issues.