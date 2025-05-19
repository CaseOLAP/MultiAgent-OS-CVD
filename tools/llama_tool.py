from pathlib import Path
from langchain.tools import Tool
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI

def create_llama_tool(name: str, description: str, index_path: str) -> Tool:
    """
    Creates a LangChain-compatible Tool that wraps a LlamaIndex query engine.

    Args:
        name (str): Name of the tool
        description (str): Description shown to the LLM
        index_path (str): Path to a pre-built LlamaIndex directory

    Returns:
        Tool: LangChain Tool usable inside an AgentExecutor
    """
    # Load saved index from storage
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    # Set global LLM for querying
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # Define a query function
    def query_fn(query: str) -> str:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return str(response)

    return Tool(
        name=name,
        description=description,
        func=query_fn
    )
