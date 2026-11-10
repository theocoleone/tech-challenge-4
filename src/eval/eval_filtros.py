"""Mostra o efeito dos filtros de metadados na recuperação (desafio adicional do briefing).

Para cada pergunta com filtro do golden set, roda a mesma busca (híbrida + rerank)
com e sem o filtro e mede no top-8:
- conformidade ao recorte: fração das evidências que pertencem ao recorte pedido
  (descritiva: com filtro é 100% por construção);
- precisão@8 de relevância, pelos julgamentos de eval/julgamentos.csv. Esta é a
  métrica que justifica os filtros: uma avaliação fora do recorte não responde à
  pergunta (uma reclamação de outra categoria não serve para "problemas de X").

Avaliações do top-8 sem filtro que ainda não foram julgadas entram na planilha
para julgamento (`--pool`); a tabela mostra a cobertura julgada.

Uso: python -m src.eval.eval_filtros [--pool]
"""
import json
import sys
from datetime import date

import pandas as pd

from src.config import ROOT
from src.eval.check_golden import carregar
from src.eval.julgamentos import escrever, ler
from src.retrieval.filters import mascara
from src.retrieval.hybrid import Buscador
from src.retrieval.rerank import reordenar

K, TOP = 30, 8


def top8(b: Buscador, pergunta: str, filtros: dict | None) -> list[str]:
    cands = b.hibrida(pergunta, K, filtros)
    return [rid for rid, _ in reordenar(pergunta, [(r.review_id, b.por_id.texto_original[r.review_id]) for r in cands])][:TOP]


def main(pool: bool = False) -> None:
    b = Buscador("titan")
    j = ler()
    rel = {q: dict(zip(g.review_id, g.relevante)) for q, g in j.groupby("pergunta_id")}
    perguntas = [p for p in carregar(ROOT / "eval" / "golden_set.jsonl") if p["tipo"] == "filtro"]
    linhas, novos = [], []
    for p in perguntas:
        f = p["filtros_esperados"]
        no_recorte = set(b.docs.review_id[mascara(b.docs, f)])
        for modo, filtros in (("com filtro", f), ("sem filtro", None)):
            ids = top8(b, p["pergunta"], filtros)
            julg = [rel.get(p["id"], {}).get(rid, "") for rid in ids]
            julgados = [x for x in julg if x]
            linhas.append({"pergunta": p["id"], "modo": modo,
                           "conformidade": sum(r in no_recorte for r in ids) / len(ids),
                           "precisao@8": sum(x == "1" for x in julgados) / len(julgados) if julgados else float("nan"),
                           "julgado@8": len(julgados) / len(ids)})
            for rid, x in zip(ids, julg):
                if not x:
                    d = b.por_id.loc[rid]
                    novos.append({"pergunta_id": p["id"], "pergunta": p["pergunta"], "review_id": rid,
                                  "nota": str(int(d.review_score)), "categoria": d.categoria_pt, "texto": d.texto_original,
                                  "relevante": "", "encontrado_por": f"{modo} (eval filtros)"})
        print(f"{p['id']}: ok", flush=True)

    df = pd.DataFrame(linhas)
    print("\npor pergunta:\n" + df.pivot(index="pergunta", columns="modo", values=["conformidade", "precisao@8"]).round(3).to_string())
    resumo = df.groupby("modo")[["conformidade", "precisao@8", "julgado@8"]].mean().round(3)
    print("\nmédia:\n" + resumo.to_string())
    if pool and novos:
        novos = pd.DataFrame(novos).drop_duplicates(["pergunta_id", "review_id"])
        escrever(pd.concat([j, novos], ignore_index=True).sort_values("pergunta_id", kind="stable"))
        print(f"\n{len(novos)} pares novos para julgar em eval/julgamentos.xlsx")
    (ROOT / "eval" / "results" / f"{date.today():%Y-%m-%d}_filtros.json").write_text(
        json.dumps({"resumo": resumo.reset_index().to_dict("records"), "por_pergunta": linhas}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main(pool="--pool" in sys.argv)
