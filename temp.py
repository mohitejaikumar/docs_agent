import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.output_parsers import RetryOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "supersecret"
)  # Use a strong secret key in production


class DatabaseChatMessageHistory(InMemoryChatMessageHistory):
    """Chat message history that stores messages in a database."""

    def __init__(self, session_id: str, db_path: str = "chat_history.db"):
        self.session_id = session_id
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._create_table()
        super().__init__()
        logging.info(
            f"DatabaseChatMessageHistory initialized for session: {self.session_id}"
        )

    def _create_table(self) -> None:
        """Create the message table if it doesn't exist."""
        try:
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS message_store (
                    session_id TEXT,
                    type TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the database."""
        super().add_message(message)
        try:
            self.cursor.execute(
                """
                INSERT INTO message_store (session_id, type, content)
                VALUES (?, ?, ?)
                """,
                (self.session_id, message.type, message.content),
            )
            self.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Error adding message: {e}")

    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from memory or database."""
        messages = super().messages
        if not messages:
            try:
                self.cursor.execute(
                    "SELECT type, content FROM message_store WHERE session_id = ? ORDER BY timestamp ASC",
                    (self.session_id,),
                )
                rows = self.cursor.fetchall()
                return [self._deserialize_message(row) for row in rows]
            except sqlite3.Error as e:
                logging.error(f"Error retrieving messages: {e}")
                return []
        return messages

    def _deserialize_message(self, row: tuple) -> BaseMessage:
        """Deserialize a message row."""
        type, content = row
        return {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}.get(
            type, BaseMessage
        )(content=content, type=type)

    def clear(self) -> None:
        """Clear messages from database and memory."""
        super().clear()
        try:
            self.cursor.execute(
                "DELETE FROM message_store WHERE session_id = ?", (self.session_id,)
            )
            self.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Error clearing messages: {e}")


class CustomCallbackHandler(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        logging.info(f"Event {name} received: {data}")


class RAGSession:
    def __init__(
        self,
        session_id,
        knowledge_base_path,
        max_token_limit=1000,
        llm_model_name="gpt-3.5-turbo",
        db_path="chat_history.db",
    ):
        self.session_id = session_id
        self.chat_history = DatabaseChatMessageHistory(session_id, db_path)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=self.chat_history,
        )
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0)
        try:
            loader = TextLoader(knowledge_base_path)
            documents = loader.load()
            texts = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(documents)
            self.db = Chroma.from_documents(texts, OpenAIEmbeddings())
            self.retriever = self.db.as_retriever()
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                callbacks=[CustomCallbackHandler()],
            )
        except Exception as e:
            logging.error(f"Error loading knowledge base: {e}")
            self.retriever = None

    def query(self, question):
        if not self.retriever:
            return "Error: No knowledge base loaded.", []
        try:
            result = self.qa({"question": question})
            return result["answer"], result["source_documents"]
        except Exception as e:
            return f"Error processing request: {e}", []


class RAGApplication:

    def __init__(self):
        self.sessions = {}

    def create_session(self, knowledge_base_path):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = RAGSession(session_id, knowledge_base_path)
        return session_id

    def get_session(self, session_id):
        return self.sessions.get(session_id)


rag_app = RAGApplication()


@app.route("/create_session", methods=["POST"])
def create_session_route():
    data = request.get_json()
    session_id = rag_app.create_session(data.get("knowledge_base_path"))
    session["session_id"] = session_id
    return jsonify({"session_id": session_id}), 201


@app.route("/query", methods=["POST"])
def query_route():
    session_id = session.get("session_id")
    question = request.get_json().get("question")
    rag_session = rag_app.get_session(session_id)
    answer, sources = (
        rag_session.query(question) if rag_session else ("No active session.", [])
    )
    return jsonify({"answer": answer, "sources": str(sources)})


if __name__ == "__main__":
    app.run(debug=True)
