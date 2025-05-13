from typing import Dict
from langchain.agents import initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.llama_tool import create_llama_tool
from memory.local_memory import get_agent_memory
from memory.global_memory import GlobalMemoryStore

# Prompt for the Pathway Agent
SYSTEM_PROMPT = """
You are the Pathway Agent. Your role is to analyze the biological summary provided by the PubMed Agent
and identify all molecular or cellular pathways involved in the mechanism.

Focus on:
- Signal transduction (e.g., MAPK, NF-κB)
- Metabolic and oxidative stress pathways
- Apoptotic or inflammatory cascades
- Pathway alterations in disease states

Use your pathway index to validate or expand the answer.
"""

# Tool: LlamaIndex search over curated pathway data (e.g., KEGG, Reactome summaries)
pathway_tool = create_llama_tool(
    name="PathwaySearchTool",
    description="Search biological pathways implicated in oxidative stress and cardiovascular disease.",
    index_path="llama_indexes/pathway"
)

# LangChain AgentExecutor
def pathway_agent_executor(memory_store: GlobalMemoryStore) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    return initialize_agent(
        tools=[pathway_tool],
        llm=llm,
        agent_type="openai-functions",
        memory=get_agent_memory("pathway", memory_store),
        verbose=True,
        agent_kwargs={"system_message": SYSTEM_PROMPT}
    )

# LangGraph-compatible callable
def pathway_agent_node(memory_store: GlobalMemoryStore) -> Runnable:
    agent = pathway_agent_executor(memory_store)

    def invoke(state: Dict) -> Dict:
        pubmed_report = state.get("pubmed_report", "")
        output = agent.invoke({"input": pubmed_report})

        memory_store.save("pathway_agent", output)

        return {
            **state,
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "pathway_agent": output
            },
            "history": state.get("history", []) + [("pathway_agent", output)]
        }

    return invoke
