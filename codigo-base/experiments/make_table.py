"""
Gera a Tabela 1 do relatório (LaTeX) a partir de experiments/results/.

Saída: experiments/tabela_resultados.tex  (colar no .tex com \\input{...} ou copiar)

Uso (a partir de codigo-base/):
    .venv/bin/python experiments/make_table.py            # usa booktabs
    .venv/bin/python experiments/make_table.py --hline    # sem pacote extra
"""
import argparse
import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")
OUT = os.path.join(BASE_DIR, "tabela_resultados.tex")

BASE = {"lr": "0.1", "epochs": "200", "hidden": "64"}   # configuração base
FINAL_RUNS = ["lr0.5_h64_ep500", "lr0.5_h256_ep500"]   # combinações finais

GROUP_NAMES = {"taxa_de_aprendizado": "$\\alpha$",
               "epocas": "Épocas", "neuronios": "$H$"}


def br(x, d):
    return f"{x:.{d}f}".replace(".", "{,}")


def value_label(group, value):
    # Valores da tabela com vírgula decimal (0,001 em vez de 0.001)
    return value.replace(".", "{,}") if group == "taxa_de_aprendizado" else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hline", action="store_true", help="usa \\hline em vez de booktabs")
    args = ap.parse_args()
    top, mid, bot = (("\\hline", "\\hline", "\\hline") if args.hline
                     else ("\\toprule", "\\midrule", "\\bottomrule"))

    with open(os.path.join(RESULTS, "summary.csv")) as f:
        rows = list(csv.DictReader(f))

    lines = [
        "% Tabela gerada por experiments/make_table.py - não editar os números à mão",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrr}",
        top,
        "Variado & Valor & Erro & Acur.\\ (\\%) & Tempo (s) \\\\",
        mid,
    ]
    groups = list(dict.fromkeys(r["grupo"] for r in rows))
    for g_i, g in enumerate(groups):
        grp = [r for r in rows if r["grupo"] == g]
        best = max(float(r["acuracia_final_teste"]) for r in grp)
        for i, r in enumerate(grp):
            is_base = (r["lr"] == BASE["lr"] and r["epochs"] == BASE["epochs"]
                       and r["hidden"] == BASE["hidden"])
            acc = float(r["acuracia_final_teste"])
            acc_s = br(100 * acc, 2)
            if acc == best:
                acc_s = f"\\textbf{{{acc_s}}}"
            name = GROUP_NAMES[g] if i == 0 else ""
            val = value_label(g, r["valor"]) + ("$^\\dagger$" if is_base else "")
            lines.append(f"{name} & {val} & {br(float(r['erro_final_treino']), 4)} & "
                         f"{acc_s} & {br(float(r['tempo_s']), 0)} \\\\")
        if g_i < len(groups) - 1:
            lines.append("\\addlinespace" if not args.hline else "\\hline")

    # Combinações finais (melhores valores de cada grupo)
    lines.append(mid)
    finals = []
    for name in FINAL_RUNS:
        with open(os.path.join(RESULTS, name, "meta.json")) as f:
            finals.append(json.load(f))
    best_final = max(m["final_test_accuracy"] for m in finals)
    for i, m in enumerate(finals):
        acc_s = br(100 * m["final_test_accuracy"], 2)
        if m["final_test_accuracy"] == best_final:
            acc_s = f"\\textbf{{{acc_s}}}"
        cfg = f"{br(m['lr'], 1)} / {m['hidden']} / {m['epochs']}"
        label = "Comb." if i == 0 else ""
        lines.append(f"{label} & {cfg} & "
                     f"{br(m['final_train_loss'], 4)} & {acc_s} & "
                     f"{br(m['total_time_s'], 0)} \\\\")
    lines += [
        bot,
        "\\end{tabular}",
        "\\caption{Erro final de treino (entropia cruzada) e acurácia final de teste "
        "para taxa de aprendizado ($\\alpha$), épocas e neurônios na camada "
        "escondida ($H$). Em cada grupo varia-se um hiperparâmetro e fixam-se os "
        "demais na configuração base ($^\\dagger$: $\\alpha=0{,}1$, $H=64$, "
        "200 épocas). As combinações finais (Comb.) estão no formato "
        "$\\alpha$\\,/\\,$H$\\,/\\,épocas. Em negrito, a maior acurácia de cada grupo.}",
        "\\label{tab:resultados}",
        "\\end{table}",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSalvo em {OUT}")
    if not args.hline:
        print("Requer \\usepackage{booktabs} no preâmbulo (ou rode com --hline).")


if __name__ == "__main__":
    main()
