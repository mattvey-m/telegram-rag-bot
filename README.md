# 📚 Telegram RAG Bot

A Telegram bot that answers questions based on your PDF documents using Retrieval-Augmented Generation (RAG). Upload your papers or documents — the bot will find relevant information and answer questions about them.

## How It Works

1. PDF documents are loaded and split into chunks
2. Chunks are embedded using `sentence-transformers` and stored in a FAISS vector index
3. When a user asks a question, the most relevant chunks are retrieved
4. A local LLM (via Ollama) generates an answer based only on the retrieved context

```
User question → Embedding → FAISS search → Top chunks → LLM → Answer
```

## Features

- Answers questions strictly from your documents (no hallucinations from general knowledge)
- Cites source PDF and page number for each answer
- Persistent vector store — no need to re-index on every restart
- `/stats`, `/reload`, `/help` commands for basic management
- Supports multiple PDFs in a folder

## Tech Stack

| Component | Library |
|---|---|
| Telegram bot | `python-telegram-bot` |
| PDF loading | `langchain`, `pypdf` |
| Embeddings | `sentence-transformers` (MiniLM-L6-v2) |
| Vector store | `FAISS` |
| LLM | `Ollama` (llama3.1:8b) |
| RAG chain | `LangChain` |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally with `llama3.1:8b` pulled

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/telegram-rag-bot.git
cd telegram-rag-bot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the LLM model via Ollama**
```bash
ollama pull llama3.1:8b
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Fill in your values in .env
```

**5. Add your PDF files**

Place your PDF documents into the `./pdfs` folder (or the path you set in `PDF_FOLDER_PATH`).

**6. Run the bot**
```bash
python telegram_rag_bot_env.py
```

On first run, the bot will index your PDFs and save the vector store to `faiss_index/`. Subsequent runs will load the existing index automatically.

## Environment Variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from [@BotFather](https://t.me/BotFather) |
| `PDF_FOLDER_PATH` | Path to the folder with PDF files (default: `./pdfs`) |

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | Usage instructions |
| `/stats` | Number of indexed chunks and status |
| `/reload` | Re-index PDFs from the folder |

## Project Structure

```
telegram-rag-bot/
├── telegram_rag_bot_env.py   # Main bot logic
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Files excluded from git
└── pdfs/                      # Place your PDF documents here
```
