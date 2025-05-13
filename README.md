
# OS-CVD Multi-Agent Explorer

A LangChain + LlamaIndex powered multi-agent system to investigate the **biological, molecular, and pharmacological associations** between **Oxidative Stress (OS)** and **Cardiovascular Disease (CVD)**.

---

## 📌 Project Architecture

```

User Query
↓
PubMed Agent (literature search)
↓
\[Protein Agent | Pathway Agent | Drug Agent]  ← Fan-out
↓
Summarizer Agent (final scientific report)

```

Each agent is autonomous, uses domain-specific vector tools, and works with LLM-based reasoning and memory.

---

## 🧠 Agents

| Agent Name       | Role                                                             |
|------------------|------------------------------------------------------------------|
| `pubmed-agent`   | Queries biomedical literature to extract ROS–CVD associations    |
| `protein-agent`  | Identifies relevant proteins and their functional roles          |
| `pathway-agent`  | Maps involved signaling or metabolic pathways                    |
| `drug-agent`     | Extracts drugs related to ROS/CVD mechanisms                     |
| `summarizer-agent` | Synthesizes all agent outputs into a final scientific report   |

---

## 📁 Folder Structure

```

ros-cvd-multiagent/
├── agents/                # LangChain-compatible agent nodes
├── tools/                 # LlamaIndex tool + index builder
├── memory/                # Global + per-agent memory
├── llama\_indexes/         # FAISS-based LlamaIndex indexes
├── data/                  # Chunked documents for agents
├── raw-docs/              # Raw files to build indexes from
├── orchestrator.py        # Executes routing logic
├── run-simulation.py      # CLI entry point
├── requirements.txt
└── README.md

````

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Add Raw Docs

Put `.txt`, `.pdf`, or `.md` files under:

```
raw-docs/pubmed/
raw-docs/protein/
raw-docs/pathway/
raw-docs/drug/
```

### 3. Build Indexes

```bash
python tools/index_builder.py
```

### 4. Run the System

```bash
python run-simulation.py
```

---

## ✅ Example Query

```
How does ROS contribute to cardiomyopathy and what proteins, pathways, and drugs are involved?
```

---

## 📜 License

MIT License — free to use, modify, and extend.

