"""
Gera as figuras do relatório a partir de experiments/results/*/history.csv.

Saída em experiments/figures/ (PDF vetorial para o LaTeX + PNG para conferência).

Regras do template AAAI-27 (CameraReady2027.tex) atendidas aqui:
  - texto dentro das figuras com no mínimo 9 pt: as figuras são geradas com a
    largura FINAL (3,0 in ~ 0.9\\columnwidth) e fonte de 9 pt, sem recorte
    automático, para que não sejam reduzidas ao entrar no documento.
    Incluir com: \\includegraphics[width=0.9\\columnwidth]{fig_xxx}
  - legível em tons de cinza e por daltônicos: cada série tem estilo de linha
    próprio + rótulo direto; paleta validada (sem amarelo, que some em cinza)
  - nenhuma fonte embutida no PDF: o texto é convertido em contornos com o
    Ghostscript (o template proíbe Type 3 e fontes CID/Identity-H)

Uso (a partir de codigo-base/):
    .venv/bin/python experiments/make_figures.py
"""
import csv
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(BASE_DIR, "figures")

WIDTH = 3.0  # polegadas = largura final no documento (0.9 * 3,3 in)
FONT = 9     # mínimo exigido pelo template para texto em figuras

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # azul, laranja, verde-água, violeta
STYLES = ["-", "--", "-.", ":"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
    "ytick.labelsize": FONT, "legend.fontsize": FONT, "legend.title_fontsize": FONT,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def br(value, decimals):
    """Número com vírgula decimal (padrão em português)."""
    return f"{value:.{decimals}f}".replace(".", ",")


BR_TICKS = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def dodge(values, min_gap):
    """Afasta verticalmente rótulos próximos demais, preservando a ordem."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    shift = min(0.0, 100 - pos[-1])          # não passa do topo do eixo (100%)
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def load(name):
    with open(os.path.join(RESULTS, name, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["epoch"]) for r in rows],
            [float(r["train_loss"]) for r in rows],
            [100 * float(r["test_accuracy"]) for r in rows])


def style_axes(ax_l, ax_a):
    ax_l.set_ylabel("Erro de treino")
    ax_a.set_ylabel("Acurácia de teste (%)")
    ax_a.set_xlabel("Época")
    ax_a.set_ylim(0, 100)
    ax_l.set_ylim(bottom=0)
    ax_l.yaxis.set_major_formatter(BR_TICKS)


def comparison(runs, legend_title, filename, max_epoch):
    """Painéis empilhados (erro de treino e acurácia de teste), legenda no topo
    e rótulo direto com a acurácia final no fim de cada curva."""
    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(WIDTH, 4.0), sharex=True)
    ends = []
    for i, (label, name) in enumerate(runs):
        ep, loss, acc = load(name)
        ep, loss, acc = ep[:max_epoch], loss[:max_epoch], acc[:max_epoch]
        kw = dict(color=COLORS[i], linestyle=STYLES[i], label=label)
        ax_l.plot(ep, loss, **kw)
        ax_a.plot(ep, acc, **kw)
        ends.append((ep[-1], acc[-1]))
    for (x, a), y in zip(ends, dodge([a for _, a in ends], min_gap=9.5)):
        ax_a.annotate(f"{br(a, 1)}%", (x, a), xytext=(x + max_epoch * 0.03, y),
                      textcoords="data", va="center", color=INK, annotation_clip=False)
    style_axes(ax_l, ax_a)
    fig.legend(*ax_l.get_legend_handles_labels(), title=legend_title, ncol=4,
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=0.8, handletextpad=0.4,
               borderaxespad=0.1)
    # Margens fixas (sem bbox_inches="tight"): a figura sai com exatamente WIDTH
    fig.subplots_adjust(left=0.19, right=0.80, top=0.83, bottom=0.11, hspace=0.12)
    fig.align_ylabels()
    save(fig, filename)


def epochs_figure(filename):
    """Rodada base de 500 épocas com os pontos 50/100/200/500 marcados."""
    ep, loss, acc = load("lr0.1_h64_ep500")
    marks = [50, 100, 200, 500]
    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(WIDTH, 3.6), sharex=True)
    ax_l.plot(ep, loss, color=COLORS[0])
    ax_a.plot(ep, acc, color=COLORS[0])
    for m in marks:
        for ax, ys in ((ax_l, loss), (ax_a, acc)):
            ax.plot(m, ys[m - 1], "o", ms=5, color=COLORS[0], mec="white", mew=0.8)
        last = m == marks[-1]
        ax_a.annotate(f"{br(acc[m - 1], 1)}%", (m, acc[m - 1]),
                      xytext=(-2 if last else 6, -14), textcoords="offset points",
                      ha="right" if last else "left", color=INK)
    style_axes(ax_l, ax_a)
    ax_a.set_xticks([0, 100, 200, 300, 400, 500])
    fig.subplots_adjust(left=0.19, right=0.95, top=0.97, bottom=0.13, hspace=0.12)
    fig.align_ylabels()
    save(fig, filename)


def best_figure(name, filename):
    """Curvas de erro e de acurácia da melhor configuração (empilhadas)."""
    ep, loss, acc = load(name)
    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(WIDTH, 3.4), sharex=True)
    ax_l.plot(ep, loss, color=COLORS[0])
    ax_a.plot(ep, acc, color=COLORS[0])
    style_axes(ax_l, ax_a)
    ax_l.annotate(f"{br(loss[-1], 3)}", (ep[-1], loss[-1]), xytext=(0, 6),
                  textcoords="offset points", ha="right", color=INK)
    ax_a.annotate(f"{br(acc[-1], 2)}%", (ep[-1], acc[-1]), xytext=(0, -14),
                  textcoords="offset points", ha="right", color=INK)
    fig.subplots_adjust(left=0.19, right=0.95, top=0.97, bottom=0.14, hspace=0.12)
    fig.align_ylabels()
    save(fig, filename)


def outline_fonts(pdf_path):
    """O matplotlib grava fontes como CID TrueType/Identity-H, que o template
    AAAI-27 proíbe mesmo dentro de figuras ("must be converted to outlines").
    O Ghostscript converte todo o texto do PDF em contornos (curvas)."""
    tmp = pdf_path + ".tmp.pdf"
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dNoOutputFonts",
                    "-sDEVICE=pdfwrite", f"-sOutputFile={tmp}", pdf_path], check=True)
    os.replace(tmp, pdf_path)


def save(fig, filename):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{filename}.{ext}"))
    plt.close(fig)
    outline_fonts(os.path.join(FIGURES, f"{filename}.pdf"))
    print("  ", filename)


if __name__ == "__main__":
    print("Gerando figuras em", FIGURES)
    comparison([("0,001", "lr0.001_h64_ep200"), ("0,01", "lr0.01_h64_ep200"),
                ("0,1", "lr0.1_h64_ep500"), ("0,5", "lr0.5_h64_ep200")],
               "Taxa de aprendizado ($H$ = 64)", "fig_taxa", max_epoch=200)
    comparison([("4", "lr0.1_h4_ep200"), ("16", "lr0.1_h16_ep200"),
                ("64", "lr0.1_h64_ep500"), ("256", "lr0.1_h256_ep200")],
               "Neurônios $H$ (taxa 0,1)", "fig_neuronios", max_epoch=200)
    epochs_figure("fig_epocas")
    best_figure("lr0.5_h256_ep500", "fig_melhor")
