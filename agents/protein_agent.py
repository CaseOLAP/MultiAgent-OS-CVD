from typing import Dict
from langchain.agents import initialize_agent, AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.llama_tool import create_llama_tool
from memory.local_memory import get_agent_memory
from memory.global_memory import GlobalMemoryStore

# System prompt for the Protein Agent
SYSTEM_PROMPT = """
You are the Protein Agent. Your job is to analyze a literature summary provided by the PubMed Agent
and extract all relevant proteins, enzymes, or structural molecules mentioned or implied.

For each identified protein:
- Explain its function and role in the ROS/CVD context
- Highlight localization or expression details if known
- Provide any association with disease or therapy

Use your protein literature index to enrich your answers.
"""

# ✅ Create the Protein semantic search tool from LlamaIndex
protein_tool = create_llama_tool(
    name="ProteinSearchTool",
    description="Use this tool to search protein roles, structures, and functions in cardiovascular or oxidative stress contexts.",
    index_path="llama_indexes/protein"
)

# ✅ Create the LangChain agent executor
def protein_agent_executor(memory_store: GlobalMemoryStore) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    return initialize_agent(
        tools=[protein_tool],
        llm=llm,
        agent_type="openai-functions",  # Uses OpenAI function-calling interface
        memory=get_agent_memory("protein", memory_store),
        verbose=True,
        agent_kwargs={"system_message": SYSTEM_PROMPT}
    )

# ✅ LangGraph-compatible callable agent node
def protein_agent_node(memory_store: GlobalMemoryStore) -> Runnable:
    agent = protein_agent_executor(memory_store)

    def invoke(state: Dict) -> Dict:
        pubmed_report = state.get("pubmed_report", "")
        output = agent.invoke({"input": pubmed_report})

        # Save to global memory
        memory_store.save("protein_agent", output)

        return {
            **state,
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "protein_agent": output
            },
            "protein_report": output,
            "history": state.get("history", []) + [("protein_agent", output)]
        }

    return invoke
