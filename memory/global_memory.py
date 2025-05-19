from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import Dict

class GlobalMemoryStore:
    def __init__(self):
        # Stores final output from each agent
        self.memory_data: Dict[str, str] = {}
        # Tracks local chat history for each agent
        self.chat_histories: Dict[str, InMemoryChatMessageHistory] = {}

    def save(self, agent_name: str, output: str):
        """Save output from an agent."""
        self.memory_data[agent_name] = output

    def load(self, agent_name: str) -> str:
        """Retrieve output from an agent."""
        return self.memory_data.get(agent_name, "")

    def get_local_chat_memory(self, agent_name: str) -> InMemoryChatMessageHistory:
        """Get or create a local chat memory instance."""
        if agent_name not in self.chat_histories:
            self.chat_histories[agent_name] = InMemoryChatMessageHistory()
        return self.chat_histories[agent_name]

    def get_all_outputs(self) -> Dict[str, str]:
        """Returns all stored agent outputs."""
        return self.memory_data.copy()
