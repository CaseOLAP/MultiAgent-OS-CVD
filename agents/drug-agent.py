from typing import Dict
from langchain.agents import initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import Runnable
from tools.llama_tool import create_llama_tool
from memory.local_memory import get_agent_memory
from memory.global_memory import GlobalMemoryStore

# System prompt for the Drug Agent
SYSTEM_PROMPT = """
You are the Drug Agent. Based on the biological summary from the PubMed Agent,
your job is to identify drugs that influence or modulate the disease mechanism.

Focus on:
- Drugs that induce or reduce oxidative stress
- Cardiovascular therapeutics relevant to the context
- Drug-protein or drug-pathway interactions
- Any known adverse effects or indications

Use your drug index to support your claims.
"""

# Tool: semantic drug search tool
drug_tool = create_llama_tool(
    name="DrugSearchTool",
    description="Search pharmacological compounds related to ROS, protein targets, or CVD pathways.",
    index_path="llama_indexes/drug"
)

# LangChain agent executor
def drug_agent_executor(memory_store: GlobalMemoryStore) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    return initialize_agent(
        tools=[drug_tool],
        llm=llm,
        agent_type="openai-functions",
        memory=get_agent_memory("drug", memory_store),
        verbose=True,
        agent_kwargs={"system_message": SYSTEM_PROMPT}
    )

# Callable node
def drug_agent_node(memory_store: GlobalMemoryStore) -> Runnable:
    agent = drug_agent_executor(memory_store)

    def invoke(state: Dict) -> Dict:
        pubmed_report = state.get("pubmed_report", "")
        output = agent.invoke({"input": pubmed_report})

        memory_store.save("drug_agent", output)

        return {
            **state,
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "drug_agent": output
            },
            "history": state.get("history", []) + [("drug_agent", output)]
        }

    return invoke
