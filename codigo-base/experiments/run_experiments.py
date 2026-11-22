"""
Executa os experimentos da Seção 4 do enunciado sem passar pela interface web,
registrando perda de treino e acurácia de teste em TODAS as épocas.

Por que não usar o server.py: ele só grava losses.csv/accuracies.csv a cada
100 épocas (curvas com 2 pontos em 200 épocas; nenhum em 50 épocas).

Fidelidade ao server.py: cada época faz exatamente o que server.gradient_descent_step()
faz: (1) learn.gradient_descent_step no conjunto de treino inteiro (full-batch),
que devolve a perda ANTES da atualização; (2) acurácia no conjunto de teste
com os pesos JÁ atualizados.

Uso (a partir de codigo-base/):
    .venv/bin/python experiments/run_experiments.py            # grade da Seção 4
    .venv/bin/python experiments/run_experiments.py --only 0.5:256:500
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "src"))
import learn  # noqa: E402

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Configuração base: fixa enquanto um hiperparâmetro varia
BASE = {"lr": 0.1, "epochs": 200, "hidden": 64}

# Seção 4 do enunciado
GRID = {
    "lr": [0.001, 0.01, 0.1, 0.5],
    "epochs": [50, 100, 200, 500],
    "hidden": [4, 16, 64, 256],
}


def run_name(lr, hidden, epochs):
    return f"lr{lr}_h{hidden}_ep{epochs}"


def train(data, lr, hidden, epochs, seed=42):
    """Treina uma RNA do zero e devolve o histórico por época."""
    X_train, X_test, y_train, y_test = data
    np.random.seed(seed)  # mesma inicialização para todas as rodadas
    parameters = learn.initialize_parameters(784, hidden, 10)

    losses, accuracies, times = [], [], []
    start = time.time()
    for epoch in range(1, epochs + 1):
        parameters, loss = learn.gradient_descent_step(X_train, y_train, parameters, lr)
        acc = learn.accuracy(y_test, learn.forward(X_test, parameters)[-1])
        losses.append(loss)
        accuracies.append(acc)
        times.append(time.time() - start)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"    época {epoch:4d}  perda {loss:.4f}  acc {acc:.4f}  ({times[-1]:.0f}s)",
                  flush=True)
    return parameters, losses, accuracies, times


def save_run(name, parameters, losses, accuracies, times, meta):
    out = os.path.join(RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "test_accuracy", "elapsed_s"])
        for i, (l, a, t) in enumerate(zip(losses, accuracies, times), start=1):
            w.writerow([i, repr(l), repr(a), f"{t:.3f}"])
    for key, arr in zip(["W1", "b1", "W2", "b2"], parameters):
        np.save(os.path.join(out, f"{key}.npy"), arr)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def planned_runs():
    """Lista de rodadas únicas (lr, hidden, max_epochs).

    O treino é full-batch e determinístico (mesma semente, mesmos dados), então
    os resultados de 50/100/200 épocas são exatamente os primeiros pontos da
    rodada de 500 épocas. Por isso o grupo "épocas" é uma única rodada de 500,
    lida nos pontos 50, 100, 200 e 500. A configuração base aparece nos três grupos.
    """
    runs = {}
    max_ep = max(GRID["epochs"])
    runs[(BASE["lr"], BASE["hidden"])] = max_ep
    for lr in GRID["lr"]:
        runs.setdefault((lr, BASE["hidden"]), BASE["epochs"])
    for h in GRID["hidden"]:
        runs.setdefault((BASE["lr"], h), BASE["epochs"])
    return [(lr, h, ep) for (lr, h), ep in runs.items()]


def write_summary(histories):
    """Tabela da Seção 4: erro final de treino e acurácia final de teste."""
    rows = []

    def add(group, value, lr, h, ep):
        hist = histories[(lr, h)]
        rows.append({
            "grupo": group, "valor": value, "lr": lr, "hidden": h, "epochs": ep,
            "erro_final_treino": hist["losses"][ep - 1],
            "acuracia_final_teste": hist["accuracies"][ep - 1],
            "melhor_acuracia_teste": max(hist["accuracies"][:ep]),
            "tempo_s": hist["times"][ep - 1],
        })

    for lr in GRID["lr"]:
        add("taxa_de_aprendizado", lr, lr, BASE["hidden"], BASE["epochs"])
    for ep in GRID["epochs"]:
        add("epocas", ep, BASE["lr"], BASE["hidden"], ep)
    for h in GRID["hidden"]:
        add("neuronios", h, BASE["lr"], h, BASE["epochs"])

    path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nResumo (Seção 4)")
    print(f"{'grupo':22s} {'valor':>7s} {'erro treino':>12s} {'acc teste':>10s} {'tempo':>8s}")
    for r in rows:
        print(f"{r['grupo']:22s} {r['valor']:>7} {r['erro_final_treino']:12.4f} "
              f"{r['acuracia_final_teste']:10.4f} {r['tempo_s']:7.0f}s")
    print(f"\nTabela salva em {path}")


def load_history(name):
    with open(os.path.join(RESULTS_DIR, name, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    return {"losses": [float(r["train_loss"]) for r in rows],
            "accuracies": [float(r["test_accuracy"]) for r in rows],
            "times": [float(r["elapsed_s"]) for r in rows]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="uma rodada avulsa no formato lr:hidden:epochs")
    parser.add_argument("--force", action="store_true", help="refaz rodadas já salvas")
    args = parser.parse_args()

    if args.only:
        lr, h, ep = args.only.split(":")
        runs = [(float(lr), int(h), int(ep))]
    else:
        runs = planned_runs()

    print("Carregando MNIST...", flush=True)
    t = time.time()
    data = learn.load_mnist()
    print(f"  ok ({time.time() - t:.0f}s): treino {data[0].shape}, teste {data[1].shape}")

    histories = {}
    for i, (lr, h, ep) in enumerate(runs, start=1):
        name = run_name(lr, h, ep)
        done = os.path.exists(os.path.join(RESULTS_DIR, name, "history.csv"))
        if done and not args.force:
            print(f"[{i}/{len(runs)}] {name}: já existe, reaproveitando")
            histories[(lr, h)] = load_history(name)
            continue
        print(f"[{i}/{len(runs)}] {name}", flush=True)
        params, losses, accs, times = train(data, lr, h, ep)
        meta = {"lr": lr, "hidden": h, "epochs": ep, "seed": 42,
                "final_train_loss": losses[-1], "final_test_accuracy": accs[-1],
                "best_test_accuracy": max(accs),
                "best_epoch": int(np.argmax(accs)) + 1, "total_time_s": times[-1]}
        save_run(name, params, losses, accs, times, meta)
        histories[(lr, h)] = {"losses": losses, "accuracies": accs, "times": times}

    if not args.only:
        write_summary(histories)


if __name__ == "__main__":
    main()
