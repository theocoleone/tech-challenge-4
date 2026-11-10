"""Calibra o limiar de "sem evidência" no conjunto de calibração.

Para cada pergunta de eval/calibracao.jsonl (20 com evidência, 20 sem), roda a
análise da pergunta, a busca e o rerank (sem gerar resposta) e guarda a maior
nota do rerank. O rerank compara as avaliações com a busca reescrita (só o
assunto), como na chain: com a pergunta inteira ("Quais reclamações existem
sobre...") as notas ficavam baixas até para perguntas com evidência. O limiar escolhido é o que melhor separa os dois grupos (maior F1
da recusa). O golden set não é usado aqui: ele fica só para o teste final.

Também mede, no nível do par (pergunta, avaliação), a nota do rerank das
avaliações julgadas relevantes e não relevantes em eval/julgamentos.csv.

Uso: python -m src.eval.calibrar
"""
import json
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT
from src.eval.check_golden import carregar
from src.eval.julgamentos import ler
from src.generation.chain import K_CANDIDATOS, _buscador
from src.retrieval.query_transform import analisar
from src.retrieval.rerank import reordenar


def top_rerank(pergunta: str) -> float:
    b = _buscador()
    a = analisar(pergunta)
    filtros = a.filtros.como_dict()
    candidatos = b.hibrida(a.consulta_reescrita, K_CANDIDATOS, filtros)
    texto = b.por_id.texto_original
    ordenados = reordenar(a.consulta_reescrita, [(r.review_id, texto[r.review_id]) for r in candidatos])
    return ordenados[0][1] if ordenados else 0.0


def melhor_limiar(scores: np.ndarray, sem_evidencia: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Recusa quando top-1 < limiar. Positivo = pergunta sem evidência."""
    linhas = []
    for limiar in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        recusa = scores < limiar
        vp = int((recusa & sem_evidencia).sum()); fp = int((recusa & ~sem_evidencia).sum())
        fn = int((~recusa & sem_evidencia).sum())
        prec = vp / (vp + fp) if vp + fp else 0.0
        rec = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        linhas.append({"limiar": limiar, "precisao_recusa": prec, "recall_recusa": rec, "f1": f1,
                       "falsa_recusa": fp / max(1, (~sem_evidencia).sum())})
    curva = pd.DataFrame(linhas)
    melhor = curva[curva.f1 == curva.f1.max()]
    # entre empates, o ponto do meio do intervalo é o mais robusto
    return float(melhor.limiar.iloc[len(melhor) // 2]), curva


def main() -> None:
    perguntas = carregar(ROOT / "eval" / "calibracao.jsonl")
    res = []
    for p in perguntas:
        s = top_rerank(p["pergunta"])
        res.append({"id": p["id"], "pergunta": p["pergunta"], "sem_evidencia": p["status_esperado"] == "sem_evidencia", "top1": s})
        print(f"{p['id']} top1={s:.3f}  {p['pergunta'][:60]}", flush=True)
    df = pd.DataFrame(res)
    limiar, curva = melhor_limiar(df.top1.to_numpy(), df.sem_evidencia.to_numpy())
    ponto = curva[curva.limiar == limiar].iloc[0]

    # nível do par: notas do rerank dos julgamentos
    j = ler()
    pares = []
    for (pid, pergunta), g in j.groupby(["pergunta_id", "pergunta"]):
        notas = dict(reordenar(analisar(pergunta).consulta_reescrita, list(zip(g.review_id, g.texto))))
        pares += [{"relevante": r == "1", "score": notas[rid]} for rid, r in zip(g.review_id, g.relevante)]
    pares = pd.DataFrame(pares)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(curva.limiar, curva.f1, label="F1 da recusa")
    ax[0].plot(curva.limiar, curva.falsa_recusa, label="taxa de falsa recusa")
    ax[0].axvline(limiar, ls="--", c="gray"); ax[0].set_xlabel("limiar (maior nota do rerank)"); ax[0].legend(); ax[0].set_title("Calibração do limiar")
    ax[1].hist([df[~df.sem_evidencia].top1, df[df.sem_evidencia].top1], bins=20, range=(0, 1), label=["com evidência", "sem evidência"])
    ax[1].axvline(limiar, ls="--", c="gray"); ax[1].set_xlabel("maior nota do rerank"); ax[1].legend(); ax[1].set_title("Perguntas de calibração")
    plt.tight_layout(); plt.savefig(ROOT / "docs" / "imagens" / "curva_limiar.png", dpi=120)

    saida = {"data": f"{date.today():%Y-%m-%d}", "limiar": limiar, "n_com_evidencia": int((~df.sem_evidencia).sum()),
             "n_sem_evidencia": int(df.sem_evidencia.sum()), "no_limiar": ponto.round(3).to_dict(),
             "pares_relevantes_mediana": float(pares[pares.relevante].score.median()),
             "pares_nao_relevantes_mediana": float(pares[~pares.relevante].score.median()),
             "pares_relevantes_acima_do_limiar": float((pares[pares.relevante].score >= limiar).mean()),
             "pares_nao_relevantes_acima_do_limiar": float((pares[~pares.relevante].score >= limiar).mean()),
             "perguntas": res}
    (ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval" / "results" / f"{date.today():%Y-%m-%d}_calibracao_limiar.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float))
    print(f"\nlimiar = {limiar} | {ponto.round(3).to_dict()}")
    print({k: round(v, 3) for k, v in saida.items() if k.startswith("pares")})


if __name__ == "__main__":
    main()
