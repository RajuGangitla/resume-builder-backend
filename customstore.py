from langchain.schema import BaseChatMessageHistory
from resume_builder import Resume
from typing import List
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)


# Simple in-memory custom store
class CustomStore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomStore, cls).__new__(cls)
            cls._instance.store = {}
        return cls._instance

    def get_or_create_session(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = {
                "messages": [],
                "resume_data": {
                    "personal_section": {},
                    "experience_section": [],
                    "education_section": [],
                    "projects_section": [],
                    "skills_section": {
                        "languages": [],
                        "frameworks": [],
                        "developer_tools": [],
                        "libraries": []
                    }
                }
            }
        return self.store[session_id]

    def add_message(self, session_id, message):
        session = self.get_or_create_session(session_id)
        session["messages"].append(message)

    def get_messages(self, session_id):
        session = self.get_or_create_session(session_id)
        return session["messages"]

    def update_resume(self, session_id, resume_data):
        session = self.get_or_create_session(session_id)
        session["resume_data"] = resume_data

    def get_resume(self, session_id):
        session = self.get_or_create_session(session_id)
        resume_object = Resume()
        resume_object.resume_data = session["resume_data"]
        return resume_object

# Custom chat message history compatible with LangChain
class CustomChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, ttl=None):
        self.session_id = session_id
        self.store = CustomStore()
        self.ttl = ttl  # Not implemented, kept for compatibility

    @property
    def messages(self):
        """Retrieve messages, converting to BaseMessage objects, newest first."""
        stored_messages = self.store.get_messages(self.session_id)
        # Handle legacy format and ensure compatibility
        formatted_items = []
        for msg in stored_messages[::-1]:  # Newest first
            if isinstance(msg, dict):
                # Fix legacy format with "role"
                if "role" in msg and "type" not in msg:
                    msg_type = "human" if msg["role"] == "user" else "ai"
                    formatted_items.append({"type": msg_type, "content": msg["content"]})
                else:
                    formatted_items.append(msg)  # Already in message_to_dict format
            else:
                formatted_items.append(message_to_dict(msg))  # Convert BaseMessage if needed
        return messages_from_dict(formatted_items)

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message, storing as a dict."""
        msg_dict = message_to_dict(message)  # {"type": "human" or "ai", "content": "..."}
        self.store.add_message(self.session_id, msg_dict)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add multiple messages."""
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """Clear session messages."""
        session = self.store.get_or_create_session(self.session_id)
        session["messages"] = []