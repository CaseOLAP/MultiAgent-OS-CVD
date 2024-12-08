# Context-Aware Multi-Agent AI System to Explore the Oxidative Stress in Cardiovascular Medicine

## Overview
This repository contains the implementation of a **context-aware multi-agent AI system** designed to explore the complex interplay between **oxidative stress (OS)** and **cardiovascular diseases (CVDs)**. The system leverages advanced AI methodologies, including **knowledge graph (KG) construction**, **graph neural networks (GNNs)** for link prediction, and a modular **multi-agent framework** to dynamically validate and refine insights. The objective is to bridge the gap between fragmented biomedical data and actionable discoveries, accelerating research in cardiovascular medicine.

---

## Features
- **Comprehensive Knowledge Graph (KG):**
  - Integrates biomedical data from PubMed, UniProt, DrugBank, and Reactome.
  - Models nodes (proteins, pathways, drugs, diseases) and edges (relationships) with high fidelity.
  
- **Graph Neural Network (GNN):**
  - Implements state-of-the-art GNN models for predicting novel OS-CVD relationships.
  - Identifies high-confidence links between biomarkers, pathways, and drug targets.

- **Multi-Agent AI Framework:**
  - Modular architecture with specialized agents:
    - **UniProt Agent**: Protein data and functional annotations.
    - **CVD Agent**: Pathways and mechanisms underlying cardiovascular diseases.
    - **OS Agent**: Analysis of oxidative stress biomarkers and mechanisms.
    - **Drug Agent**: Drug interactions and therapeutic implications.
    - **Reactome Agent**: Systems-level analysis of metabolic and signaling pathways.
  - Central orchestrator for task management and inter-agent communication.

- **Dynamic Analysis and Refinement:**
  - Agents leverage contextual understanding to refine predictions.
  - Feedback loops ensure iterative improvement of outputs.

- **Interactive Insights Visualization:**
  - Visualizes KGs and predicted relationships via interactive dashboards.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Recommended: Google Cloud Platform (GCP) account for deploying scalable workflows.
- Required Python Libraries:
  - `tensorflow`, `torch`
  - `neo4j`, `networkx`
  - `flask`, `fastapi`
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn`, `plotly`
  - `google-cloud-*` for GCP integrations

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/context-aware-ai-system.git
   cd context-aware-ai-system
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Google Cloud services:
   - Set up a Google Cloud project and enable APIs for Vertex AI, BigQuery, Cloud Storage, and Compute Engine.
   - Authenticate using the `gcloud` CLI:
     ```bash
     gcloud auth login
     gcloud config set project [PROJECT_ID]
     ```

5. Start the application locally:
   ```bash
   python app.py
   ```

---

## Usage
### Knowledge Graph Construction
- Run `scripts/build_knowledge_graph.py` to construct the KG from biomedical datasets:
  ```bash
  python scripts/build_knowledge_graph.py --input data/ --output kgraph/
  ```

### GNN Model Training
- Use `scripts/train_gnn.py` to train the link prediction model:
  ```bash
  python scripts/train_gnn.py --graph kgraph/ --output models/
  ```

### Multi-Agent System
- Launch the multi-agent framework with:
  ```bash
  python multi_agent_system.py
  ```

### Visualization
- Visualize insights via interactive dashboards:
  ```bash
  python visualization.py
  ```

---

## Project Structure
```
context-aware-ai-system/
├── data/                   # Raw and preprocessed datasets
├── kgraph/                 # Knowledge graph files
├── models/                 # Trained GNN models
├── scripts/                # Scripts for data processing, KG construction, and GNN training
├── agents/                 # Implementation of specialized agents
├── visualization/          # Scripts for visualizing insights
├── app.py                  # Main entry point for running the system
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Contributing
We welcome contributions to this project! To contribute:
1. Fork this repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a meaningful commit message"
   ```
4. Push to your fork and submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
This work is inspired by:
- Panday et al., "Context-aware Multi-agent AI System to Explore the Oxidative Stress in Cardiovascular Medicine" (2024).
- [Panday et al., "Data-Driven Insights into the Association Between Oxidative Stress and Calcium-Regulating Proteins in Cardiovascular Disease" (2024)](https://www.mdpi.com/2076-3921/13/11/1420).
- Google Cloud's Vertex AI and Neo4j for advanced data integration and processing.

---

## Contact
For questions or collaboration opportunities, please reach out to:
- **Namuna Panday** | Department of Physiology, UCLA | namuna@g.ucla.edu
- **Dibakar Sigdel** | Department of Physiology, UCLA | sigdeldkr@gmail.com
