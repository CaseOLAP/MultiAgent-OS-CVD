import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def build_index_for_agent(agent_name: str):
    raw_dir = Path(f"raw-docs/{agent_name}")
    index_dir = Path(f"llama_indexes/{agent_name}")
    index_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[!] No documents found in {raw_dir}")
        return

    print(f"[+] Reading documents for {agent_name.upper()}...")

    # Load documents
    reader = SimpleDirectoryReader(str(raw_dir))
    documents = reader.load_data()
    print(f"[✓] Loaded {len(documents)} documents for {agent_name}")

    # Use OpenAI's cheapest models
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.chunk_size = 512

    # ✅ First-time creation: don't try to load, just create storage and persist
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Now persist to disk
    index.storage_context.persist(persist_dir=str(index_dir))
    print(f"[✓] Index built and saved to: {index_dir}")


if __name__ == "__main__":
    for agent in ["pubmed", "protein", "pathway", "drug"]:
        build_index_for_agent(agent)
