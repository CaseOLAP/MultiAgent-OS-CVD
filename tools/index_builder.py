import os
from pathlib import Path
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI

def build_index_for_agent(agent_name: str):
    raw_dir = Path(f"raw-docs/{agent_name}")
    index_dir = Path(f"llama_indexes/{agent_name}")
    index_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[!] No documents found in {raw_dir}")
        return

    print(f"[+] Reading documents for {agent_name.upper()}...")

    # Load all documents
    reader = SimpleDirectoryReader(str(raw_dir))
    documents = reader.load_data()

    print(f"[✓] Loaded {len(documents)} documents for {agent_name}")

    # Build vector index
    service_context = ServiceContext.from_defaults(llm=OpenAI(temperature=0.3))
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Save to disk
    index.save_to_disk(str(index_dir))
    print(f"[✓] Index saved to: {index_dir}")

if __name__ == "__main__":
    for agent in ["pubmed", "protein", "pathway", "drug"]:
        build_index_for_agent(agent)
