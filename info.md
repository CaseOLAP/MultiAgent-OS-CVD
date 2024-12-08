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

