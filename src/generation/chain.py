"""Pipeline RAG de Voice of Customer, montado como uma chain LCEL.

pergunta
  -> análise (escopo, tipo, filtros, busca reescrita)       [LLM]
  -> ramo A: fora do escopo                 -> recusa
     ramo D: filtro sem nenhuma avaliação   -> recusa
     busca: híbrida (Titan + BM25) -> rerank -> seleção diversa (MMR)
       ramo B: maior nota do rerank abaixo do limiar     -> sem evidência
       ramo C: poucas avaliações acima do limiar         -> evidência insuficiente
  -> contexto com as avaliações em tags <review>
  -> resposta estruturada com citações                        [LLM]
  -> validação: citações, linguagem de frequência (guard.py)
     ramo C também quando o modelo diz que não há evidência ou nenhuma citação é válida

Cada resposta gera um trace em logs/traces.jsonl.

Uso: python -m src.generation.chain "O que os clientes reclamam da entrega?"
"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from src.config import ROOT, get_llm, get_settings
from src.generation.context import montar_contexto
from src.generation.guard import limpar_ids, validar
from src.generation.schemas import RespostaVoC
from src.retrieval.diversity import mmr
from src.retrieval.filters import mascara
from src.retrieval.hybrid import Buscador
from src.retrieval.query_transform import analisar, tem_assunto_proprio
from src.retrieval.rerank import reordenar

PROMPTS = Path(__file__).resolve().parent / "prompts"
K_CANDIDATOS = 30  # candidatos da busca híbrida que vão para o rerank
K_DETALHADAS = 15  # candidatos extras só entre avaliações detalhadas
MIN_PALAVRAS_DETALHE = 10
LIMIAR_RERANK = 0.30  # calibrado em eval/calibracao.jsonl (ver eval/results/*_calibracao_limiar.json)
MIN_ACIMA_LIMIAR = 3  # abaixo disso, a evidência é considerada insuficiente
VERSAO_PROMPTS = "analise_v1+resposta_v1"
N_CONTEXTO = 8     # avaliações que vão para o LLM
LAMBDA_MMR = 0.5   # peso da relevância contra a diversidade na seleção
PESO_DETALHE = 0.3 # bônus para avaliações mais detalhadas (ver _buscar)


@lru_cache
def _buscador() -> Buscador:
    return Buscador("titan")


@lru_cache
def _gerador():
    prompt = ChatPromptTemplate.from_messages([
        ("system", (PROMPTS / "resposta_v1.md").read_text()),
        ("human", "{contexto}\n\nPergunta do gestor: {pergunta}"),
    ])
    return prompt | get_llm(estruturado=RespostaVoC, papel="geracao")


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
    # Segunda busca só entre avaliações detalhadas: sem ela, os candidatos ficam cheios de
    # frases curtas ("Atraso na entrega") e a resposta não tem o que contar.
    detalhadas = b.hibrida(a.consulta_reescrita, K_DETALHADAS, {**estado["filtros"], "n_palavras_min": MIN_PALAVRAS_DETALHE})
    vistos = {r.review_id for r in candidatos}
    candidatos += [r for r in detalhadas if r.review_id not in vistos]
    texto = b.por_id.texto_original
    # o rerank compara com a busca reescrita (só o assunto); a pergunta original vai para a geração
    ordenados = reordenar(a.consulta_reescrita, [(r.review_id, texto[r.review_id]) for r in candidatos])
    ids, scores = [rid for rid, _ in ordenados], [sc for _, sc in ordenados]
    # Avaliações com mais detalhe rendem respostas mais concretas: o rerank sozinho
    # prefere frases curtas e literais ("Atraso na entrega"). Um bônus pequeno pelo
    # tamanho (até 25 palavras) desempata a favor de quem explica o que aconteceu.
    palavras = b.por_id.loc[ids, "n_palavras"].clip(upper=25).to_numpy() / 25 if ids else []
    pesos = [sc + PESO_DETALHE * w for sc, w in zip(scores, palavras)]
    escolhidos = mmr(ids, pesos, b.vetores(ids), N_CONTEXTO, LAMBDA_MMR) if ids else []
    escolhidos.sort(key=lambda i: scores[i], reverse=True)  # no contexto, do mais ao menos relevante
    estado["tempos"]["busca"] = time.time() - t
    return {**estado, "selecionados": [ordenados[i] for i in escolhidos]}


def _gerar(estado: dict) -> dict:
    sel = estado["selecionados"]
    acima = [rid for rid, sc in sel if sc >= LIMIAR_RERANK]
    # Pergunta sem assunto próprio ("problemas nas avaliações nota 1 de X"): o recorte
    # vem dos filtros e toda avaliação dele é evidência; o limiar de relevância do rerank
    # (que compara com uma busca genérica) não se aplica.
    so_recorte = bool(sel) and bool(estado["filtros"]) and not tem_assunto_proprio(estado["analise"])
    if not so_recorte:
        if not sel or sel[0][1] < LIMIAR_RERANK:
            return _saida(estado, "sem_evidencia", "Não encontrei avaliações que tratem desse assunto na base "
                          "(nenhuma passou do limiar de relevância).", [])
        if len(acima) < MIN_ACIMA_LIMIAR:
            return _saida(estado, "evidencia_fraca", f"Só {len(acima)} avaliação(ões) tratam desse assunto, o que é "
                          "pouco para tirar uma conclusão. Elas estão listadas nas evidências.",
                          [{"afirmacao": "Avaliações encontradas sobre o assunto (sem generalização).", "review_ids": acima}])

    t = time.time()
    b = _buscador()
    ids = [rid for rid, _ in sel]
    entrada = {"contexto": montar_contexto(ids, b.por_id, estado["filtros"]), "pergunta": estado["pergunta"]}
    resposta = _gerador().invoke(entrada)
    v = validar(resposta.resposta, [i.model_dump() for i in resposta.insights], set(ids))
    estado["validacao"] = {"citacoes_invalidas": v.citacoes_invalidas, "frequencia": v.linguagem_de_frequencia, "refeita": False}
    if v.linguagem_de_frequencia and estado["analise"].tipo == "especifica":
        aviso = (f"\n\nAtenção: a resposta anterior usou {v.linguagem_de_frequencia}. Estas avaliações são uma amostra "
                 "da busca, não a base inteira: reescreva sem termos de frequência nem percentuais.")
        resposta = _gerador().invoke({**entrada, "pergunta": entrada["pergunta"] + aviso})
        v = validar(resposta.resposta, [i.model_dump() for i in resposta.insights], set(ids))
        estado["validacao"].update(refeita=True, frequencia_depois=v.linguagem_de_frequencia)
    estado["tempos"]["geracao"] = time.time() - t
    if not resposta.tem_evidencia or not v.insights:
        return _saida(estado, "evidencia_fraca", limpar_ids(resposta.resposta), v.insights)
    return _saida(estado, "ok", limpar_ids(resposta.resposta), v.insights)


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
    saida = {"trace_id": uuid.uuid4().hex[:12], "pergunta": estado["pergunta"], "status": status,
             "resposta": resposta, "insights": insights, "evidencias": evidencias, "tipo": a.tipo,
             "pergunta_reescrita": a.consulta_reescrita, "filtros_aplicados": estado["filtros"],
             "n_no_recorte": estado["n_no_recorte"], "tempos": {k: round(v, 2) for k, v in estado["tempos"].items()}}
    _registrar_trace(saida, estado)
    return saida


def _registrar_trace(saida: dict, estado: dict) -> None:
    """Uma linha por resposta em logs/traces.jsonl (pasta fora do git)."""
    s = get_settings()
    registro = {
        "trace_id": saida["trace_id"], "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pergunta": saida["pergunta"], "pergunta_reescrita": saida["pergunta_reescrita"], "tipo": saida["tipo"],
        "filtros": saida["filtros_aplicados"], "n_no_recorte": saida["n_no_recorte"], "status": saida["status"],
        "selecionados": [{"review_id": e["review_id"], "score_rerank": e["score_rerank"]} for e in saida["evidencias"]],
        "citados": sorted({rid for i in saida["insights"] for rid in i["review_ids"]}),
        "validacao": estado.get("validacao"), "limiar": LIMIAR_RERANK, "prompts": VERSAO_PROMPTS,
        "modelos": {"analise": s.llm_model_id, "geracao": s.gen_model_id, "embedding": s.emb_model_id},
        "tempos": saida["tempos"],
    }
    pasta = ROOT / "logs"
    pasta.mkdir(exist_ok=True)
    with open(pasta / "traces.jsonl", "a") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


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
