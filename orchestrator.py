from agents.pubmed_agent import pubmed_agent_node
from agents.protein_agent import protein_agent_node
from agents.pathway_agent import pathway_agent_node
from agents.drug_agent import drug_agent_node
from agents.summarizer_agent import summarizer_agent_node
from memory.global_memory import GlobalMemoryStore

def run_orchestration(user_query: str) -> str:
    # Initialize shared memory
    memory_store = GlobalMemoryStore()

    # Step 1: Run PubMed Agent on user query
    print("\n🔍 Running PubMed Agent...")
    pubmed_state = {
        "user_input": user_query,
        "agent_outputs": {},
        "history": []
    }
    state_after_pubmed = pubmed_agent_node(memory_store)(pubmed_state)

    # Step 2: Fan out to Protein, Pathway, Drug Agents (parallel in logic)
    print("🧬 Running Protein Agent...")
    state_with_protein = protein_agent_node(memory_store)(state_after_pubmed)

    print("🧪 Running Pathway Agent...")
    state_with_pathway = pathway_agent_node(memory_store)(state_with_protein)

    print("💊 Running Drug Agent...")
    state_with_drug = drug_agent_node(memory_store)(state_with_pathway)

    # Step 3: Run Summarizer Agent
    print("\n📝 Generating final summary...")
    final_state = summarizer_agent_node(memory_store)(state_with_drug)

    return final_state["final_summary"]
