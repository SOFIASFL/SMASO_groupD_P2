# SMASO_groupD_P2
# Agent-Based Market Simulator with LLM Integration

This project implements a stochastic financial market simulation using the Mesa framework. It models a single-asset market driven by Geometric Brownian Motion (GBM) dynamics and integrates a Large Language Model (LLM) agent to provide fundamental trading analysis.

## Project Overview

The simulation consists of a social network of investors and a centralized market mechanism. One of the contributions of this implementation is the **AnalystLLMAgent**, which leverages the Groq API (specifically the Llama-3.3-70b model) to interpret numerical market data and output structured trading recommendations (Buy, Sell, or Hold) along with confidence levels and textual reasoning. The analyst signal is logged as natural language text for interpretability, while the structured recommendation (Plan) is used internally by investor agents for decision-making.

## Configuration

To function correctly, the Analyst Agent requires a valid API Key from Groq. The system uses environment variables to load this credential securely.

**Step-by-step configuration:**

1. Obtain an API Key from the Groq console.
2. Create a file named `.env` in the root directory of this project (at the same level as `run_smoke_test.py`).
3. Add the following line to the `.env` file:
```text
GROQ_API_KEY=your_api_key_here

```



**Note:** The LLM-based analyst is an optional component. When disabled or unavailable, the simulation can be executed in a deterministic fallback mode, ensuring full reproducibility of the core agent-based model.

```bash
pip install mesa openai python-dotenv networkx numpy

```

## Usage

To validate the integration between the market model and the (optional) LLM component, a smoke test script is provided. This script initializes the market environment and performs a system check of the (optional) LLM integration to ensure the agent is generating valid decisions before running the simulation loop.


**On Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "src"; python run_smoke_test.py
```

## Structure

* **src/market/environment.py**: Defines the market physics using Geometric Brownian Motion (GBM).
* **src/agents/analyst.py**: Implements the `AnalystLLMAgent` class, handling API communication and prompt engineering.
* **src/agents/investor.py**: Implements the investor agents that operate within the market.
* **src/mesa_model/model.py**: The central Mesa model that orchestrates the agents and the environment.
* **run_smoke_test.py**: Execution script for validation and testing.
