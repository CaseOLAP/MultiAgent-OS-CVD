from langchain.tools import Tool
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

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
    # Load index from disk
    index = VectorStoreIndex.load_from_disk(index_path)

    # Define a query function
    def query_fn(query: str) -> str:
        service_context = ServiceContext.from_defaults(llm=OpenAI(temperature=0.3))
        query_engine = index.as_query_engine(service_context=service_context)
        response = query_engine.query(query)
        return str(response)

    return Tool(
        name=name,
        description=description,
        func=query_fn
    )
