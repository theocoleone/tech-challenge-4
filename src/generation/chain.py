"""Pipeline RAG de Voice of Customer, montado como uma chain LCEL.

pergunta
  -> análise (escopo, tipo, filtros, busca reescrita)       [LLM]
  -> ramo: fora do escopo  -> recusa
           filtro vazio    -> recusa ("nenhuma avaliação nesse recorte")
           busca           -> híbrida (Titan + BM25) -> rerank -> seleção diversa (MMR)
  -> contexto com as avaliações em tags <review>
  -> resposta estruturada com citações                        [LLM]

Uso: python -m src.generation.chain "O que os clientes reclamam da entrega?"
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from src.config import get_llm
from src.generation.context import montar_contexto
from src.generation.schemas import RespostaVoC
from src.retrieval.diversity import mmr
from src.retrieval.filters import mascara
from src.retrieval.hybrid import Buscador
from src.retrieval.query_transform import analisar
from src.retrieval.rerank import reordenar

PROMPTS = Path(__file__).resolve().parent / "prompts"
K_CANDIDATOS = 30  # candidatos da busca híbrida que vão para o rerank
N_CONTEXTO = 8     # avaliações que vão para o LLM
LAMBDA_MMR = 0.5   # peso da relevância contra a diversidade na seleção


@lru_cache
def _buscador() -> Buscador:
    return Buscador("titan")


@lru_cache
def _gerador():
    prompt = ChatPromptTemplate.from_messages([
        ("system", (PROMPTS / "resposta_v1.md").read_text()),
        ("human", "{contexto}\n\nPergunta do gestor: {pergunta}"),
    ])
    return prompt | get_llm(estruturado=RespostaVoC)


def _analisar(estado: dict) -> dict:
    t = time.time()
    a = analisar(estado["pergunta"])
    filtros = a.filtros.como_dict()
    b = _buscador()
    return {**estado, "analise": a, "filtros": filtros,
            "n_no_recorte": int(mascara(b.docs, filtros).sum()), "tempos": {"analise": time.time() - t}}


def _buscar(estado: dict) -> dict:
    t = time.time()
    b = _buscador()
    a = estado["analise"]
    candidatos = b.hibrida(a.consulta_reescrita, K_CANDIDATOS, estado["filtros"])
    texto = b.por_id.texto_original
    ordenados = reordenar(estado["pergunta"], [(r.review_id, texto[r.review_id]) for r in candidatos])
    ids, scores = [rid for rid, _ in ordenados], [sc for _, sc in ordenados]
    escolhidos = mmr(ids, scores, b.vetores(ids), N_CONTEXTO, LAMBDA_MMR) if ids else []
    escolhidos.sort(key=lambda i: scores[i], reverse=True)  # no contexto, do mais ao menos relevante
    estado["tempos"]["busca"] = time.time() - t
    return {**estado, "selecionados": [ordenados[i] for i in escolhidos]}


def _gerar(estado: dict) -> dict:
    t = time.time()
    b = _buscador()
    ids = [rid for rid, _ in estado["selecionados"]]
    resposta = _gerador().invoke({"contexto": montar_contexto(ids, b.por_id), "pergunta": estado["pergunta"]})
    estado["tempos"]["geracao"] = time.time() - t
    return _saida(estado, "ok" if resposta.tem_evidencia else "sem_evidencia", resposta.resposta,
                  [i.model_dump() for i in resposta.insights])


def _recusa(status: str, mensagem: str):
    return RunnableLambda(lambda estado: _saida(estado, status, mensagem, []))


def _saida(estado: dict, status: str, resposta: str, insights: list) -> dict:
    b = _buscador()
    evidencias = []
    for rid, score in estado.get("selecionados", []):
        d = b.por_id.loc[rid]
        evidencias.append({"review_id": rid, "texto": d.texto_original, "nota": int(d.review_score),
                           "categoria": d.categoria_pt, "mes_compra": int(d.ano_mes_compra),
                           "entregue_com_atraso": bool(d.entregue_com_atraso), "score_rerank": round(score, 3)})
    a = estado["analise"]
    return {"pergunta": estado["pergunta"], "status": status, "resposta": resposta, "insights": insights,
            "evidencias": evidencias, "tipo": a.tipo, "pergunta_reescrita": a.consulta_reescrita,
            "filtros_aplicados": estado["filtros"], "n_no_recorte": estado["n_no_recorte"],
            "tempos": {k: round(v, 2) for k, v in estado["tempos"].items()}}


def montar_chain():
    return RunnableLambda(_analisar) | RunnableBranch(
        (lambda e: e["analise"].escopo == "fora_escopo",
         _recusa("fora_escopo", "Essa pergunta está fora do escopo: a base contém apenas avaliações de clientes "
                                "sobre compras na Olist (2016 a 2018).")),
        (lambda e: e["n_no_recorte"] == 0,
         _recusa("filtro_sem_resultado", "Não há nenhuma avaliação com comentário no recorte pedido "
                                         "(confira os filtros aplicados).")),
        RunnableLambda(_buscar) | RunnableLambda(_gerar),
    )


@lru_cache
def chain():
    return montar_chain()


def responder(pergunta: str) -> dict:
    return chain().invoke({"pergunta": pergunta})


if __name__ == "__main__":
    r = responder(" ".join(sys.argv[1:]) or "O que os clientes relatam sobre atraso na entrega?")
    print(f"\n[{r['status']}] {r['resposta']}\n")
    for i in r["insights"]:
        print(f"  - {i['afirmacao']}  {i['review_ids']}")
    print(f"\nfiltros: {r['filtros_aplicados']} | busca: \"{r['pergunta_reescrita']}\" | tempos: {r['tempos']}")
    print("\nevidências:")
    for e in r["evidencias"]:
        print(f"  [{e['score_rerank']:+.2f}] nota {e['nota']} | {e['categoria']} | {e['texto'][:90]!r}")
    if "--json" in sys.argv:
        print(json.dumps(r, ensure_ascii=False, indent=1))
