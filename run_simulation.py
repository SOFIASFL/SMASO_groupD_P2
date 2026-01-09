import pandas as pd
import time
import os
from src.core.config import SimConfig
from src.network.topology import build_network
from src.mesa_model.model import MarketModel
from src.agents.analyst import AnalystLLMAgent
from src.core.types import ActionType

def main():
    # --- CONFIGURAÇÃO ---
    N_STEPS = 100
    PAUSE_SEC = 1.0
    OUTPUT_FILE = "resultados_simulacao.csv"

    print(f"Starting simulation of {N_STEPS} steps...")

    # 1. Configurar Modelo
    cfg = SimConfig(seed=42, n_investors=30, topology="small_world", p=0.1, k=4)
    G = build_network(n=cfg.n_investors, topology=cfg.topology, seed=cfg.seed, p=cfg.p, k=cfg.k)
    model = MarketModel(G=G, n_investors=cfg.n_investors, seed=cfg.seed)

    # 2. Injetar Agente IA
    model.analyst = AnalystLLMAgent(unique_id=999, model=model)

    history = []

    # 3. Loop da Simulação
    for i in range(N_STEPS):
        # A. Acordar o Analista (IA)
        obs = model.analyst.observe()
        plan = model.analyst.plan(obs, "")
        model.analyst.act(plan)

        # B. Avançar o Mercado (Investidores tomam decisões aqui)
        model.step()

        # C. Recolher Dados
        current_price = model.market.price
        market_return = model.market.last_return

        # D. O que disse a IA?
        analyst_action = "WAIT"
        analyst_conf = 0.0
        analyst_source = "N/A"

        if model.analyst.recommendation:
            rec = model.analyst.recommendation
            analyst_action = rec.intended_action.name if hasattr(rec.intended_action, "name") else str(rec.intended_action)
            analyst_conf = rec.confidence
            analyst_source = rec.meta.get("source", "unknown")

        # E. O que fizeram os Investidores? (CONTAGEM NOVA)
        # Vamos contar quantos Buy/Sell houve neste turno
        n_buys = 0
        n_sells = 0
        # model.last_actions guarda a última ação de cada investidor
        if hasattr(model, "last_actions"):
            actions = list(model.last_actions.values())
            n_buys = actions.count(ActionType.BUY)
            n_sells = actions.count(ActionType.SELL)

        print(f"Step {i+1}/{N_STEPS} | Price: {current_price:.2f} | AI: {analyst_action} | Investors: {n_buys} BUYs vs {n_sells} SELLs")

        history.append({
            "Step": i,
            "Price": current_price,
            "Return": market_return,
            "Analyst_Action": analyst_action,
            "Analyst_Confidence": analyst_conf,
            "Analyst_Source": analyst_source,
            "Investors_Buy_Count": n_buys,  # Coluna Nova
            "Investors_Sell_Count": n_sells # Coluna Nova
        })

        time.sleep(PAUSE_SEC)

    # 4. Gravar
    df = pd.DataFrame(history)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Simulation finished. Results saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()