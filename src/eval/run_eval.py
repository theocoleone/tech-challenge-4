"""Roda o golden set de ponta a ponta: ramos, citações, fidelidade e números.

Três tabelas:
1. ramos — precisão e recall de cada ramo (ok, sem evidência, evidência fraca,
   filtro sem resultado, fora do escopo) e a decisão binária responder x recusar;
2. geração — validade das citações e fidelidade (cada afirmação é sustentada
   pelas avaliações citadas?, julgado por um modelo diferente do gerador);
3. agregadas — exatidão numérica: todo número da resposta saiu da contagem.

Uso: python -m src.eval.run_eval [--sem-juiz]
"""
import sys
import time
from datetime import date

import pandas as pd

from src.config import ROOT
from src.eval.check_golden import carregar
from src.eval.judge import fidelidade
from src.eval.json_utils import salvar_json
from src.generation.chain import _buscador, responder

RAMOS = ["fora_escopo", "filtro_sem_resultado", "sem_evidencia", "evidencia_fraca"]


def main(com_juiz: bool = True) -> None:
    perguntas = carregar(ROOT / "eval" / "golden_set.jsonl")
    linhas, respostas = [], []
    for p in perguntas:
        t = time.time()
        r = responder(p["pergunta"])
        ids_ev = {e["review_id"] for e in r["evidencias"]}
        citados = [rid for i in r["insights"] for rid in i["review_ids"]]
        val = r.get("validacao") or {}
        linhas.append({"id": p["id"], "tipo": p["tipo"], "esperado": p["status_esperado"], "obtido": r["status"],
                       "acertou": r["status"] == p["status_esperado"], "n_citacoes": len(citados),
                       "citacoes_validas": all(c in ids_ev for c in citados) if citados else None,
                       "citacoes_removidas": val.get("citacoes_invalidas", 0),
                       "numeros_fora_da_contagem": len(val.get("numeros_fora_da_contagem") or []),
                       "segundos": round(time.time() - t, 1)})
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
    com_citacoes = df[df.n_citacoes > 0]
    taxa_citacoes = com_citacoes.citacoes_validas.astype(bool).mean() if len(com_citacoes) else float("nan")
    print(f"citações válidas: {taxa_citacoes:.1%} (n={len(com_citacoes)} respostas com citações) | "
          f"tempo médio: {df.segundos.mean():.1f}s")
    print("\nmatriz (linhas = esperado, colunas = obtido):")
    print(pd.crosstab(df.esperado, df.obtido).to_string())

    # tabela 2: geração
    por_id = _buscador().por_id
    fid, vereditos = (fidelidade([r for r in respostas if r["status"] == "ok"], por_id) if com_juiz
                      else (float("nan"), []))
    geracao = pd.Series({
        "respostas com evidência": int((df.obtido == "ok").sum()),
        "respostas com citações": len(com_citacoes),
        "citações válidas": round(taxa_citacoes, 3),
        "citações removidas pelo guard": int(df.citacoes_removidas.sum()),
        "afirmações julgadas": len(vereditos),
        "fidelidade (afirmações sustentadas)": round(fid, 3) if vereditos else float("nan"),
    }, name="valor")
    print("\ngeração:\n" + geracao.to_frame().to_string())

    # tabela 3: perguntas agregadas
    ids_agreg = set(df.loc[df.tipo == "agregada", "id"])
    respostas_com_contagem = [
        r for r in respostas if r["id"] in ids_agreg and r.get("contagem")
    ]
    respostas_agreg = [
        r for r in respostas
        if (
            r["id"] in ids_agreg
            and r["tipo"] == "agregada"
            and r["status"] == "ok"
            and r.get("contagem")
            and isinstance(r.get("validacao"), dict)
        )
    ]
    ids_agreg_validas = {r["id"] for r in respostas_agreg}
    linhas_agreg = df[df.id.isin(ids_agreg_validas)]
    tab_agreg = pd.Series({
        "perguntas agregadas": len(ids_agreg),
        "responderam com contagem": len(respostas_com_contagem),
        "respostas agregadas válidas": len(linhas_agreg),
        "exatidão numérica (sem número fora da contagem)": round(
            (linhas_agreg.numeros_fora_da_contagem == 0).mean(), 3
        ) if len(linhas_agreg) else float("nan"),
        "base média (avaliações contadas)": int(pd.Series([r["contagem"]["total_com_tema"] for r in respostas_agreg]).mean())
        if respostas_agreg else 0,
    }, name="valor")
    print("\nperguntas agregadas:\n" + tab_agreg.to_frame().to_string())

    saida = ROOT / "eval" / "results" / f"{date.today():%Y-%m-%d}_golden_ramos.json"
    salvar_json(saida, {"resumo_por_ramo": tab.reset_index().to_dict("records"),
                        "geracao": geracao.to_dict(), "agregadas": tab_agreg.to_dict(),
                        "fidelidade_vereditos": vereditos, "por_pergunta": linhas,
                        "respostas": respostas})
    print(f"\nsalvo em {saida.relative_to(ROOT)}")


if __name__ == "__main__":
    main(com_juiz="--sem-juiz" not in sys.argv)
