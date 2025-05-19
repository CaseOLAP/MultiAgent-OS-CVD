from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from memory.global_memory import GlobalMemoryStore

# ✅ LLM used for summarization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# ✅ Structured summarization prompt
SUMMARY_TEMPLATE = ChatPromptTemplate.from_template("""
You are a scientific summarizer. Based on the following inputs from specialized agents,
write a comprehensive, structured, and clear report for a biomedical researcher.

Original User Query:
{user_query}

PubMed Report:
{pubmed_report}

Protein Findings:
{protein_report}

Pathway Insights:
{pathway_report}

Drug Associations:
{drug_report}

Synthesize these into a single, well-structured summary.
Include:
- Introduction
- Mechanism overview
- Protein involvement
- Pathways involved
- Drug modulation
- Conclusion
""")

# ✅ LangGraph-compatible summarizer node
def summarizer_agent_node(memory_store: GlobalMemoryStore) -> Runnable:

    def invoke(state: Dict) -> Dict:
        # Extract all inputs from prior agents
        query = state.get("user_input", "")
        agent_outputs = state.get("agent_outputs", {})

        pubmed_report = agent_outputs.get("pubmed_agent", "")
        protein_report = agent_outputs.get("protein_agent", "")
        pathway_report = agent_outputs.get("pathway_agent", "")
        drug_report = agent_outputs.get("drug_agent", "")

        # Construct and send the prompt
        messages = SUMMARY_TEMPLATE.format_messages(
            user_query=query,
            pubmed_report=pubmed_report,
            protein_report=protein_report,
            pathway_report=pathway_report,
            drug_report=drug_report
        )

        response = llm(messages)
        summary_text = response.content

        # Save to global memory
        memory_store.save("summarizer_agent", summary_text)

        return {
            **state,
            "final_summary": summary_text,
            "history": state.get("history", []) + [("summarizer_agent", summary_text)]
        }

    return invoke
