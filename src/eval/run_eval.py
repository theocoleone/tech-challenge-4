"""Roda o golden set de ponta a ponta e mede se cada pergunta cai no ramo certo.

Ramos: ok (responde com evidência), sem_evidencia, evidencia_fraca,
filtro_sem_resultado, fora_escopo. Para cada ramo de recusa, mede precisão e
recall; para as perguntas com evidência, a taxa de falsa recusa e a validade
das citações. As respostas completas ficam em eval/results/.

Uso: python -m src.eval.run_eval
"""
import json
import time
from datetime import date

import pandas as pd

from src.config import ROOT
from src.eval.check_golden import carregar
from src.generation.chain import responder

RAMOS = ["fora_escopo", "filtro_sem_resultado", "sem_evidencia", "evidencia_fraca"]


def main() -> None:
    perguntas = carregar(ROOT / "eval" / "golden_set.jsonl")
    linhas, respostas = [], []
    for p in perguntas:
        t = time.time()
        r = responder(p["pergunta"])
        ids_ev = {e["review_id"] for e in r["evidencias"]}
        citados = [rid for i in r["insights"] for rid in i["review_ids"]]
        linhas.append({"id": p["id"], "tipo": p["tipo"], "esperado": p["status_esperado"], "obtido": r["status"],
                       "acertou": r["status"] == p["status_esperado"], "n_citacoes": len(citados),
                       "citacoes_validas": all(c in ids_ev for c in citados), "segundos": round(time.time() - t, 1)})
        respostas.append({"id": p["id"], **r})
        print(f"{p['id']} esperado={p['status_esperado']:22s} obtido={r['status']:22s} {'ok' if linhas[-1]['acertou'] else 'ERRO'}", flush=True)

    df = pd.DataFrame(linhas)
    print(f"\nacerto do ramo: {df.acertou.mean():.1%} ({df.acertou.sum()}/{len(df)})")
    por_ramo = []
    for ramo in RAMOS + ["ok"]:
        esperados, obtidos = df.esperado == ramo, df.obtido == ramo
        vp = int((esperados & obtidos).sum())
        por_ramo.append({"ramo": ramo, "n_esperado": int(esperados.sum()), "n_obtido": int(obtidos.sum()),
                         "precisao": vp / obtidos.sum() if obtidos.sum() else float("nan"),
                         "recall": vp / esperados.sum() if esperados.sum() else float("nan")})
    tab = pd.DataFrame(por_ramo).set_index("ramo").round(3)
    print(tab.to_string())
    recusas = {"sem_evidencia", "evidencia_fraca", "fora_escopo", "filtro_sem_resultado"}
    dev_recusar, recusou = df.esperado.isin(recusas), df.obtido.isin(recusas)
    print(f"\ndecisão responder x recusar: {(dev_recusar == recusou).mean():.1%} | "
          f"recusas corretas: {(dev_recusar & recusou).sum()}/{dev_recusar.sum()} | "
          f"respondeu quando devia recusar: {(dev_recusar & ~recusou).sum()}")
    com_ev = df[df.esperado == "ok"]
    print(f"\nfalsa recusa nas perguntas com evidência: {(com_ev.obtido != 'ok').mean():.1%} (n={len(com_ev)})")
    print(f"citações válidas: {df.citacoes_validas.mean():.1%} | tempo médio: {df.segundos.mean():.1f}s")
    print("\nmatriz (linhas = esperado, colunas = obtido):")
    print(pd.crosstab(df.esperado, df.obtido).to_string())

    saida = ROOT / "eval" / "results" / f"{date.today():%Y-%m-%d}_golden_ramos.json"
    saida.write_text(json.dumps({"resumo_por_ramo": tab.reset_index().to_dict("records"), "por_pergunta": linhas,
                                 "respostas": respostas}, ensure_ascii=False, indent=1, default=float))
    print(f"\nsalvo em {saida.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
