import os
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
RETRIEVER_K = 6


class RAGTelegramBot:
    def __init__(self, telegram_token: str, pdf_folder_path: str):
        self.telegram_token = telegram_token
        self.pdf_folder_path = pdf_folder_path
        self.vectorstore = None
        self.qa_chain = None

    def load_and_process_documents(self):
        """Load PDFs from folder, split into chunks, and create FAISS vector store."""
        logger.info(f"Loading PDFs from {self.pdf_folder_path}")

        if not os.path.exists(self.pdf_folder_path):
            raise FileNotFoundError(f"PDF folder not found: {self.pdf_folder_path}")

        # Load all PDFs from the folder recursively
        loader = DirectoryLoader(
            self.pdf_folder_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError(f"No PDF documents found in {self.pdf_folder_path}")

        logger.info(f"Loaded {len(documents)} document pages")

        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")

        # Create embeddings and build the vector store
        logger.info("Creating embeddings... This may take a while.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        logger.info("Vector store created successfully")

        # Save to disk so we don't have to re-index on next run
        self.vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"Vector store saved to '{FAISS_INDEX_PATH}'")

    def load_existing_vectorstore(self):
        """Load a previously saved FAISS index from disk."""
        logger.info("Loading existing vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")

    def setup_qa_chain(self):
        """Set up the RAG chain: retriever + prompt + LLM."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Run load_and_process_documents first.")

        # Prompt instructs the LLM to answer strictly from retrieved context
        prompt_template = """Use ONLY the provided context to answer the question.
Do NOT combine information from different documents.
If the context comes from multiple sources and they are inconsistent, state that explicitly.
If the answer is not fully contained in a single source, say that you cannot answer reliably.
If multiple context fragments come from the same document and are logically connected,
combine them to produce a coherent answer.

Context: {context}

Question: {question}

Detailed Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # MMR retrieval balances relevance and diversity of retrieved chunks
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "lambda_mult": 0.8}
        )

        # Local LLM via Ollama
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        logger.info("QA chain setup complete")

    # --- Telegram command handlers ---

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(
            "👋 Welcome to the Research Papers AI Assistant!\n\n"
            "I can answer questions based on the scientific papers and articles in my knowledge base.\n\n"
            "Commands:\n"
            "/start - Show this message\n"
            "/help - Get help\n"
            "/stats - Show system statistics\n"
            "/reload - Reload PDF documents\n\n"
            "Just send me your question and I'll search through the papers to find an answer!"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text(
            "ℹ️ How to use:\n\n"
            "Simply type your question about the research papers and I'll find relevant information.\n\n"
            "Examples:\n"
            "- What are the main findings about...?\n"
            "- Explain the methodology used for...\n"
            "- What does the research say about...?\n"
            "- Compare the approaches of...\n\n"
            "I'll provide answers based on the scientific papers in my database and cite my sources."
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command — show number of indexed chunks and current status."""
        if self.vectorstore is None:
            await update.message.reply_text("⚠️ System is not initialized yet.")
            return

        try:
            num_vectors = self.vectorstore.index.ntotal
            await update.message.reply_text(
                f"📊 System Statistics:\n\n"
                f"📄 Document chunks indexed: {num_vectors}\n"
                f"📁 PDF folder: {self.pdf_folder_path}\n"
                f"✅ Status: Ready"
            )
        except Exception as e:
            await update.message.reply_text(f"Error getting stats: {str(e)}")

    async def reload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reload command — re-index all PDFs from the folder."""
        await update.message.reply_text("🔄 Reloading PDF documents... This may take several minutes.")
        try:
            self.load_and_process_documents()
            self.setup_qa_chain()
            await update.message.reply_text("✅ Documents reloaded successfully!")
        except Exception as e:
            logger.error(f"Error reloading documents: {e}")
            await update.message.reply_text(f"❌ Error reloading documents: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming user questions and return RAG-generated answers."""
        if self.qa_chain is None:
            await update.message.reply_text(
                "⚠️ System is not ready yet. Please wait for initialization to complete."
            )
            return

        user_question = update.message.text
        logger.info(f"Question from @{update.effective_user.username}: {user_question}")

        await update.message.chat.send_action(action="typing")

        try:
            # Run the blocking RAG call in a thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                self.qa_chain.invoke,
                {"query": user_question}
            )

            answer = result['result']
            source_docs = result.get('source_documents', [])

            response = f"📚 **Answer:**\n\n{answer}\n\n"

            # Append unique source citations (up to 3)
            if source_docs:
                response += "📄 **Sources:**\n"
                seen_sources = set()
                for doc in source_docs[:3]:
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    if source not in seen_sources:
                        response += f"- {Path(source).name} (page {page})\n"
                        seen_sources.add(source)

            # Telegram has a 4096-character message limit
            if len(response) > 4096:
                for i in range(0, len(response), 4096):
                    await update.message.reply_text(response[i:i + 4096], parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error processing your question. Please try again or rephrase your question."
            )

    def run(self):
        """Initialize the vector store and start the Telegram bot."""
        # Load existing index if available, otherwise build from scratch
        if os.path.exists(FAISS_INDEX_PATH):
            logger.info("Found existing vector store, loading it...")
            try:
                self.load_existing_vectorstore()
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
                logger.info("Creating new vector store...")
                self.load_and_process_documents()
        else:
            logger.info("No existing vector store found, creating new one...")
            self.load_and_process_documents()

        self.setup_qa_chain()

        # Register handlers and start polling
        application = Application.builder().token(self.telegram_token).build()
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("reload", self.reload_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        logger.info("Bot started successfully and ready to answer questions!")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH", "./pdfs")

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

    if not os.path.exists(PDF_FOLDER_PATH):
        logger.info(f"Creating PDF folder: {PDF_FOLDER_PATH}")
        os.makedirs(PDF_FOLDER_PATH)

    bot = RAGTelegramBot(
        telegram_token=TELEGRAM_BOT_TOKEN,
        pdf_folder_path=PDF_FOLDER_PATH
    )
    bot.run()


if __name__ == "__main__":
    main()
