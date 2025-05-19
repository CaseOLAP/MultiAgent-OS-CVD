from typing import Dict
from langchain.agents import initialize_agent, AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.llama_tool import create_llama_tool
from memory.local_memory import get_agent_memory
from memory.global_memory import GlobalMemoryStore

# System prompt guiding the LLM agent
SYSTEM_PROMPT = """
You are the PubMed Agent. Your job is to explore biomedical literature and answer user queries
by synthesizing relevant knowledge. You have access to a semantic PubMed index that contains
articles, reviews, and disease associations. Retrieve the most relevant information and summarize it.
Be clear, scientific, and evidence-based.
"""

# ✅ Create the PubMed semantic search tool from LlamaIndex
pubmed_tool = create_llama_tool(
    name="PubMedSearchTool",
    description="Use this tool to search biomedical literature related to ROS, cardiovascular disease, inflammation, etc.",
    index_path="llama_indexes/pubmed"
)

# ✅ Create the LangChain agent executor
def pubmed_agent_executor(memory_store: GlobalMemoryStore) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    return initialize_agent(
        tools=[pubmed_tool],
        llm=llm,
        agent_type="openai-functions",  # This uses the function-calling agent interface
        memory=get_agent_memory("pubmed", memory_store),
        verbose=True,
        agent_kwargs={"system_message": SYSTEM_PROMPT}
    )

# ✅ LangGraph-compatible callable node
def pubmed_agent_node(memory_store: GlobalMemoryStore) -> Runnable:
    agent = pubmed_agent_executor(memory_store)

    def invoke(state: Dict) -> Dict:
        user_query = state["user_input"]
        output = agent.invoke({"input": user_query})

        # Save to global memory
        memory_store.save("pubmed_agent", output)

        return {
            **state,
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "pubmed_agent": output
            },
            "pubmed_report": output,
            "history": state.get("history", []) + [("pubmed_agent", output)]
        }

    return invoke
