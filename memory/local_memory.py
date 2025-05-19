from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from memory.global_memory import GlobalMemoryStore

def get_agent_memory(agent_name: str, memory_store: GlobalMemoryStore) -> ConversationSummaryBufferMemory:
    """
    Creates or retrieves an agent-specific memory instance using LangChain's
    summarization-based conversation memory, backed by the shared memory store.
    """
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3),
        memory_key="history",
        input_key="input",
        output_key="output",
        return_messages=True,
        chat_memory=memory_store.get_local_chat_memory(agent_name),
        ai_prefix=f"{agent_name}_response",
        human_prefix="user_input"
    )
