"""Pipeline RAG de Voice of Customer, montado como uma chain LCEL.

pergunta
  -> análise (escopo, tipo, filtros, busca reescrita)       [LLM]
  -> suporte lexical auditável para assuntos explícitos
  -> ramo agregado (pergunta de frequência): valida a existência do assunto,
     depois conta todos os temas classificados no recorte
     (src/analytics/aggregate.py) + exemplos representativos            [LLM]
  -> ramo A: fora do escopo                 -> recusa
     ramo E: comparação entre dois recortes -> evidência insuficiente
     ramo D: filtro sem nenhuma avaliação   -> recusa
     busca: híbrida (Titan + BM25) -> rerank -> seleção diversa (MMR)
       ramo B: assunto ausente                         -> sem evidência
       ramo C: somente 1 a 3 menções explícitas        -> evidência insuficiente
  -> contexto com as avaliações em tags <review>
  -> resposta estruturada com citações                        [LLM]
  -> validação: citações, linguagem de frequência (guard.py)
     ramo C também quando o modelo diz que não há evidência ou nenhuma citação é válida

Cada resposta gera um trace em logs/traces.jsonl.

Uso: python -m src.generation.chain "O que os clientes reclamam da entrega?"
"""
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from src.analytics.aggregate import MIN_AVALIACOES, Agregador
from src.config import ROOT, get_llm, get_settings
from src.generation.context import montar_contexto
from src.generation.guard import conferir_numeros, limpar_ids, validar
from src.generation.schemas import FiltroReviews, RespostaVoC
from src.retrieval.diversity import mmr
from src.retrieval.evidence import SuporteLexical, VerificadorEvidenciaLexical
from src.retrieval.filters import mascara
from src.retrieval.hybrid import Buscador
from src.retrieval.query_transform import analisar, pede_comparacao, temas_explicitos
from src.retrieval.rerank import reordenar

PROMPTS = Path(__file__).resolve().parent / "prompts"
K_CANDIDATOS = 30  # candidatos da busca híbrida que vão para o rerank
K_DETALHADAS = 15  # candidatos extras só entre avaliações detalhadas
MIN_PALAVRAS_DETALHE = 10
LIMIAR_EXISTENCIA = 0.39  # calibrado para o top-1 por pergunta, não para contar documentos
MAX_LEXICAIS_CANDIDATOS = 30
ARQUIVOS_PROMPT = ("analise_v1.md", "resposta_v1.md", "agregada_v1.md")
N_CONTEXTO = 8     # avaliações que vão para o LLM
N_CONTEXTO_MIN = 3 # mínimo para síntese quando há muitas menções no corpus
LAMBDA_MMR = 0.5   # peso da relevância contra a diversidade na seleção
PESO_DETALHE = 0.3 # bônus para avaliações mais detalhadas (ver _buscar)


@lru_cache
def _versao_prompts() -> dict[str, str]:
    """Identifica exatamente o conteúdo usado, mesmo se o nome do arquivo não mudar."""
    return {
        nome: hashlib.sha256((PROMPTS / nome).read_bytes()).hexdigest()[:12]
        for nome in ARQUIVOS_PROMPT
    }


@lru_cache
def _buscador() -> Buscador:
    return Buscador("titan")


@lru_cache
def _agregador() -> Agregador:
    return Agregador()


@lru_cache
def _verificador_evidencia() -> VerificadorEvidenciaLexical:
    return VerificadorEvidenciaLexical(_buscador().docs)


def _avaliar_suporte(estado: dict) -> dict:
    """Acrescenta um sinal lexical quando a pergunta tem assunto explícito."""
    if "_suporte_lexical" in estado or not estado.get("temas_alvo"):
        return estado
    suporte = _verificador_evidencia().avaliar(
        estado["pergunta"],
        temas_alvo=estado["temas_alvo"],
        filtros=estado["filtros"],
    )
    return {**estado, "_suporte_lexical": suporte}


def _suporte_como_dict(suporte: SuporteLexical) -> dict:
    return {
        "aplicavel": suporte.aplicavel,
        "faixa": suporte.faixa,
        "quantidade": suporte.quantidade,
        "termos": [list(grupo) for grupo in suporte.grupos_de_termos],
        "origem": suporte.origem_termos,
    }


def _avaliacoes(quantidade: int) -> str:
    return f"{quantidade} avaliação" if quantidade == 1 else f"{quantidade} avaliações"


def _menciona(quantidade: int) -> str:
    return "menciona" if quantidade == 1 else "mencionam"


def _inteiro_pt(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _percentuais_pt(texto: str) -> str:
    return re.sub(r"(?<!\d)(\d+)\.(\d+)(\s*%)", r"\1,\2\3", texto)


def _finalizar_texto_agregado(texto: str, contagem) -> str:
    """Garante apresentação brasileira e o denominador calculado pelo código."""
    texto = _percentuais_pt(limpar_ids(texto))
    valores = (contagem.total_com_tema, contagem.total_no_recorte)
    variantes = [
        {str(v), _inteiro_pt(v), f"{v:,}".replace(",", " ")}
        for v in valores
    ]
    if all(any(forma in texto for forma in formas) for formas in variantes):
        return texto
    polaridade = {
        "nota_baixa": "notas 1 ou 2",
        "nota_alta": "notas 4 ou 5",
        "todas": "notas 1, 2, 4 ou 5",
    }[contagem.polaridade]
    base = (
        f"Base da contagem: {_inteiro_pt(contagem.total_com_tema)} avaliações "
        f"classificadas ({polaridade}), entre {_inteiro_pt(contagem.total_no_recorte)} "
        "avaliações com comentário no recorte."
    )
    return f"{texto.rstrip()} {base}"


def _texto_agregado_deterministico(contagem) -> str:
    """Formata a contagem sem depender de o LLM copiar números corretamente."""
    itens = []
    for item in contagem.temas:
        quantidade = int(item["n"])
        percentual = _percentuais_pt(f'{item["pct"]}%')
        itens.append(
            f'{item["tema"]}: {_inteiro_pt(quantidade)} '
            f'{"avaliação" if quantidade == 1 else "avaliações"} ({percentual})'
        )
    resumo = "; ".join(itens)
    return _finalizar_texto_agregado(
        f"Os temas mais frequentes no recorte foram {resumo}.",
        contagem,
    )


def _insights_agregados_sem_numeros_invalidos(
    insights: list[dict],
    contagem,
    filtros: dict,
) -> list[dict]:
    """Mantém somente insights já sanitizados cujos números passam pelo guard."""
    return [
        {
            **insight,
            "afirmacao": _percentuais_pt(insight["afirmacao"]),
        }
        for insight in insights
        if not conferir_numeros(insight["afirmacao"], contagem, filtros)
    ]


def _insights_agregados_deterministicos(
    contagem,
    por_tema: dict[str, list[str]],
    limite: int = 3,
) -> list[dict]:
    """Fallback citável quando nenhum insight do modelo pode ser aproveitado."""
    insights = []
    for item in contagem.temas:
        ids = list(dict.fromkeys(por_tema.get(item["tema"], [])))[:3]
        if not ids:
            continue
        insights.append({
            "afirmacao": f'As avaliações citadas exemplificam o tema "{item["tema"]}".',
            "review_ids": ids,
        })
        if len(insights) == limite:
            break
    return insights


def _amostra_ids_lexicais(suporte: SuporteLexical, b: Buscador) -> list[str]:
    """Limita o rerank, priorizando relatos detalhados entre as menções exatas."""
    ids = list(suporte.review_ids)
    if len(ids) <= MAX_LEXICAIS_CANDIDATOS:
        return ids
    dados = b.por_id.loc[ids]
    if "n_palavras" in dados:
        dados = dados.sort_values("n_palavras", ascending=False, kind="stable")
    return dados.index[:MAX_LEXICAIS_CANDIDATOS].tolist()


def _mesclar_filtros(extraidos: dict, manuais: dict | None) -> dict:
    """Mescla filtros sem manter a outra ponta extraída de um intervalo contraditório."""
    manuais = manuais or {}
    combinados = {**extraidos, **manuais}
    for minimo, maximo in (
        ("review_score_min", "review_score_max"),
        ("ano_mes_compra_min", "ano_mes_compra_max"),
        ("valor_min", "valor_max"),
    ):
        if minimo not in combinados or maximo not in combinados:
            continue
        if combinados[minimo] <= combinados[maximo]:
            continue
        if minimo in manuais and maximo not in manuais:
            combinados.pop(maximo)
        elif maximo in manuais and minimo not in manuais:
            combinados.pop(minimo)
    return FiltroReviews.model_validate(combinados).como_dict()


@lru_cache
def _gerador_agregada():
    prompt = ChatPromptTemplate.from_messages([
        ("system", (PROMPTS / "agregada_v1.md").read_text()),
        ("human", "{contexto}\n\nPergunta do gestor: {pergunta}"),
    ])
    return prompt | get_llm(estruturado=RespostaVoC, papel="geracao")


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
    filtros = _mesclar_filtros(a.filtros.como_dict(), estado.get("filtros_manuais"))
    a.filtros = FiltroReviews.model_validate(filtros)
    alvos = temas_explicitos(estado["pergunta"], a)
    a.temas_alvo = alvos
    b = _buscador()
    return {**estado, "analise": a, "filtros": filtros, "temas_alvo": alvos,
            "comparacao_nao_suportada": pede_comparacao(estado["pergunta"]),
            "n_no_recorte": int(mascara(b.docs, filtros).sum()), "tempos": {"analise": time.time() - t}}


def _buscar(estado: dict) -> dict:
    """Busca e ordenação com a pergunta original; existência com sinal independente.

    Na avaliação com o pool de julgamentos, buscar e ordenar pela pergunta original
    deu precisão@8 de 0,951, contra 0,853 usando a pergunta reescrita. Já para
    decidir se existe evidência, combinamos o top-1 contra a reescrita com a
    contagem lexical auditável dos termos explícitos. O limiar do top-1 não é
    reutilizado para estimar quantas avaliações tratam do assunto.
    """
    t = time.time()
    b = _buscador()
    a = estado["analise"]
    pergunta = estado["pergunta"]
    estado = _avaliar_suporte(estado)
    suporte = estado.get("_suporte_lexical")
    candidatos = b.hibrida(pergunta, K_CANDIDATOS, estado["filtros"])
    # Segunda busca só entre avaliações detalhadas: sem ela, os candidatos ficam cheios de
    # frases curtas ("Atraso na entrega") e a resposta não tem o que contar.
    detalhadas = b.hibrida(pergunta, K_DETALHADAS, {**estado["filtros"], "n_palavras_min": MIN_PALAVRAS_DETALHE})
    ids_candidatos = [r.review_id for r in candidatos]
    vistos = set(ids_candidatos)
    ids_candidatos += [r.review_id for r in detalhadas if r.review_id not in vistos]
    vistos.update(ids_candidatos)
    if suporte and suporte.aplicavel:
        ids_candidatos += [
            rid for rid in _amostra_ids_lexicais(suporte, b)
            if rid not in vistos
        ]
    ids_candidatos = list(dict.fromkeys(ids_candidatos))
    texto = b.por_id.texto_original
    pares = [(rid, texto[rid]) for rid in ids_candidatos]
    ordenados = reordenar(pergunta, pares)
    # o portão olha também o que a busca reescrita encontra
    vistos = set(ids_candidatos)
    extras = [
        r for r in b.hibrida(a.consulta_reescrita, K_CANDIDATOS, estado["filtros"])
        if r.review_id not in vistos
    ]
    portao = dict(reordenar(a.consulta_reescrita, pares + [(r.review_id, texto[r.review_id]) for r in extras]))
    ids, scores = [rid for rid, _ in ordenados], [sc for _, sc in ordenados]
    # Avaliações com mais detalhe rendem respostas mais concretas: o rerank sozinho
    # prefere frases curtas e literais. Um bônus pequeno pelo tamanho (até 25 palavras)
    # desempata a favor de quem explica o que aconteceu.
    palavras = b.por_id.loc[ids, "n_palavras"].clip(upper=25).to_numpy() / 25 if ids else []
    pesos = [sc + PESO_DETALHE * w for sc, w in zip(scores, palavras)]
    escolhidos = mmr(ids, pesos, b.vetores(ids), N_CONTEXTO, LAMBDA_MMR) if ids else []
    escolhidos.sort(key=lambda i: scores[i], reverse=True)  # no contexto, do mais ao menos relevante
    # Em perguntas com assunto explícito, o MMR não pode forçar avaliações quase
    # irrelevantes só para completar oito itens. Mantém diversidade entre as que
    # ainda têm uma fração material da relevância do melhor resultado.
    if estado["temas_alvo"] and len(escolhidos) > N_CONTEXTO_MIN:
        corte = max(0.05, scores[0] * 0.1)
        acima_corte = [i for i in escolhidos if scores[i] >= corte]
        escolhidos = (
            acima_corte
            if len(acima_corte) >= N_CONTEXTO_MIN
            else list(range(min(N_CONTEXTO_MIN, len(ids))))
        )
    estado["tempos"]["busca"] = time.time() - t
    return {**estado, "selecionados": [ordenados[i] for i in escolhidos],
            "portao_top1": max(portao.values(), default=0.0),
            "scores_busca": dict(ordenados), "portao": portao}


def _gerar(estado: dict) -> dict:
    sel = estado["selecionados"]
    suporte = estado.get("_suporte_lexical")
    # Pergunta sem assunto próprio ("problemas nas avaliações nota 1 de X"): o recorte
    # vem dos filtros e toda avaliação dele é evidência; o limiar de existência
    # (que compara com uma busca genérica) não se aplica.
    so_recorte = bool(sel) and bool(estado["filtros"]) and not estado["temas_alvo"]
    if not so_recorte:
        if suporte and suporte.faixa == "evidencia_fraca":
            scores = estado.get("scores_busca", {})
            poucas = [(rid, scores.get(rid, 0.0)) for rid in suporte.review_ids]
            estado = {**estado, "selecionados": poucas}
            return _saida(
                estado,
                "evidencia_fraca",
                f"Encontrei apenas {_avaliacoes(suporte.quantidade)} que "
                f"{_menciona(suporte.quantidade)} "
                "explicitamente esse assunto. Isso é pouco para sintetizar um padrão.",
                [{
                    "afirmacao": "Menções explícitas encontradas, sem generalização.",
                    "review_ids": [rid for rid, _ in poucas],
                }],
            )
        suporte_suficiente = bool(suporte and suporte.faixa == "evidencia_suficiente")
        if not sel or (not suporte_suficiente and estado["portao_top1"] < LIMIAR_EXISTENCIA):
            estado = {**estado, "selecionados": []}
            return _saida(
                estado,
                "sem_evidencia",
                "Não encontrei avaliações que sustentem esse assunto na base.",
                [],
            )

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
        estado["validacao"].update(
            refeita=True,
            citacoes_invalidas=v.citacoes_invalidas,
            frequencia_depois=v.linguagem_de_frequencia,
        )
    estado["tempos"]["geracao"] = time.time() - t
    if v.linguagem_de_frequencia and estado["analise"].tipo == "especifica":
        return _saida(
            estado,
            "evidencia_fraca",
            "A síntese automática continuou generalizando uma amostra da busca e foi bloqueada.",
            [],
        )
    if not resposta.tem_evidencia:
        status = "sem_evidencia" if suporte and suporte.faixa == "sem_evidencia" else "evidencia_fraca"
        if status == "sem_evidencia":
            estado = {**estado, "selecionados": []}
        return _saida(estado, status, limpar_ids(resposta.resposta), v.insights)
    if not v.insights:
        return _saida(estado, "evidencia_fraca", limpar_ids(resposta.resposta), [])
    return _saida(estado, "ok", limpar_ids(resposta.resposta), v.insights)


def _agregar(estado: dict) -> dict:
    """Ramo agregado: os números vêm da contagem, e o LLM só redige.

    Antes de contar, uma pergunta com assunto explícito ("principais problemas
    com Pix") verifica as menções no recorte. Sem avaliações sobre o assunto,
    contar temas responderia outra pergunta.
    """
    t = time.time()
    ag = _agregador()
    b = _buscador()
    alvos = estado["temas_alvo"]
    if alvos:
        estado = _avaliar_suporte(estado)
        suporte = estado["_suporte_lexical"]
        if suporte.faixa == "sem_evidencia":
            estado = {**estado, "selecionados": []}
            return _saida(
                estado,
                "sem_evidencia",
                "Não encontrei avaliações que mencionem esse assunto no recorte; "
                "por isso não calculei uma frequência.",
                [],
            )
        if suporte.faixa == "evidencia_fraca":
            estado = {
                **estado,
                "selecionados": [(rid, None) for rid in suporte.review_ids],
            }
            return _saida(
                estado,
                "evidencia_fraca",
                f"O recorte contém apenas {_avaliacoes(suporte.quantidade)} que "
                f"{_menciona(suporte.quantidade)} explicitamente o assunto. "
                "A amostra é pequena demais "
                "para calcular uma frequência confiável.",
                [{
                    "afirmacao": "Menções explícitas encontradas, sem generalização.",
                    "review_ids": list(suporte.review_ids),
                }],
            )

    c = ag.contar(estado["pergunta"], estado["filtros"], alvos or None)
    if c.assunto_nao_modelado:
        estado["tempos"]["agregacao"] = time.time() - t
        return _saida(
            estado,
            "evidencia_fraca",
            "Há avaliações sobre o assunto, mas o catálogo de temas não o separa com "
            f"confiabilidade ({c.assunto_nao_modelado}). Por isso não calculei uma frequência.",
            [],
        )
    if c.total_com_tema < MIN_AVALIACOES:
        estado["tempos"]["agregacao"] = time.time() - t
        return _saida(
            estado,
            "evidencia_fraca",
            f"O recorte tem só {_avaliacoes(c.total_com_tema)} classificadas em tema, "
            "o que é pouco para falar do que é mais frequente.",
            [],
        )
    por_tema = {
        tema["tema"]: ag.representativas(
            tema["tema"], estado["filtros"], polaridade=c.polaridade
        )
        for tema in c.temas
    }
    ids = list(dict.fromkeys(rid for ids in por_tema.values() for rid in ids))
    # no ramo agregado as avaliações ilustram cada tema, não vêm de um ranking: sem score
    estado = {**estado, "selecionados": [(rid, None) for rid in ids], "contagem": c}
    entrada = {"contexto": ag.contexto(c, por_tema, b.por_id), "pergunta": estado["pergunta"]}
    resposta = _gerador_agregada().invoke(entrada)
    v = validar(resposta.resposta, [i.model_dump() for i in resposta.insights], set(ids))
    texto_validado = "\n".join([resposta.resposta] + [i.afirmacao for i in resposta.insights])
    numeros = conferir_numeros(texto_validado, c, estado["filtros"])
    estado["validacao"] = {
        "citacoes_invalidas": v.citacoes_invalidas,
        "numeros_fora_da_contagem": numeros,
        "refeita": False,
        "fallback_deterministico": False,
    }

    # Tenta uma correção controlada. Se ela ainda falhar, a parte numérica é
    # formatada pelo código e somente insights sanitizados são aproveitados.
    if not resposta.tem_evidencia or numeros or v.citacoes_invalidas or not v.insights:
        aviso = (
            "\n\nA saída anterior não passou pela validação. Reescreva usando exatamente "
            "os números de <contagem>; cada percentual deve vir de lá e todo insight "
            "precisa citar apenas IDs presentes nos respectivos <exemplos>."
        )
        resposta = _gerador_agregada().invoke({**entrada, "pergunta": entrada["pergunta"] + aviso})
        v = validar(resposta.resposta, [i.model_dump() for i in resposta.insights], set(ids))
        texto_validado = "\n".join([resposta.resposta] + [i.afirmacao for i in resposta.insights])
        numeros = conferir_numeros(texto_validado, c, estado["filtros"])
        estado["validacao"].update(
            citacoes_invalidas=v.citacoes_invalidas,
            numeros_fora_da_contagem=numeros,
            refeita=True,
        )

    estado["tempos"]["agregacao"] = time.time() - t
    if not resposta.tem_evidencia or numeros or v.citacoes_invalidas or not v.insights:
        insights = _insights_agregados_sem_numeros_invalidos(
            v.insights,
            c,
            estado["filtros"],
        )
        if not insights:
            insights = _insights_agregados_deterministicos(c, por_tema)
        if insights:
            estado["validacao"].update(
                fallback_deterministico=True,
                numeros_rejeitados_pelo_guard=numeros,
                numeros_fora_da_contagem=[],
            )
            return _saida(
                estado,
                "ok",
                _texto_agregado_deterministico(c),
                insights,
            )
        return _saida(
            estado,
            "evidencia_fraca",
            "A síntese automática não passou pela validação e não havia exemplos "
            "citáveis suficientes para construir uma resposta segura.",
            [],
        )
    insights = [
        {**insight, "afirmacao": _percentuais_pt(insight["afirmacao"])}
        for insight in v.insights
    ]
    return _saida(
        estado,
        "ok",
        _finalizar_texto_agregado(resposta.resposta, c),
        insights,
    )


def _recusa(status: str, mensagem: str):
    return RunnableLambda(lambda estado: _saida(estado, status, mensagem, []))


def _saida(estado: dict, status: str, resposta: str, insights: list) -> dict:
    b = _buscador()
    evidencias = []
    for rid, score in estado.get("selecionados", []):
        d = b.por_id.loc[rid]
        evidencias.append({"review_id": rid, "texto": d.texto_original, "nota": int(d.review_score),
                           "categoria": d.categoria_pt, "mes_compra": int(d.ano_mes_compra),
                           "entregue": bool(d.entregue),
                           "entregue_com_atraso": bool(d.entregue_com_atraso),
                           "score_rerank": None if score is None else round(score, 3)})
    a = estado["analise"]
    saida = {"trace_id": uuid.uuid4().hex[:12], "pergunta": estado["pergunta"], "status": status,
             "resposta": resposta, "insights": insights, "evidencias": evidencias, "tipo": a.tipo,
             "pergunta_reescrita": a.consulta_reescrita, "filtros_aplicados": estado["filtros"],
             "n_no_recorte": estado["n_no_recorte"], "tempos": {k: round(v, 2) for k, v in estado["tempos"].items()}}
    if "_suporte_lexical" in estado:
        saida["suporte_lexical"] = _suporte_como_dict(estado["_suporte_lexical"])
    if "validacao" in estado:
        saida["validacao"] = estado["validacao"]
    if "contagem" in estado:
        c = estado["contagem"]
        saida["contagem"] = {"polaridade": c.polaridade, "total_no_recorte": c.total_no_recorte,
                             "total_com_tema": c.total_com_tema, "macros": c.macros, "temas": c.temas}
    _registrar_trace(saida, estado)
    return saida


def _registrar_trace(saida: dict, estado: dict) -> None:
    """Uma linha por resposta em logs/traces.jsonl (pasta fora do git)."""
    s = get_settings()
    registro = {
        "trace_id": saida["trace_id"], "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pergunta": saida["pergunta"], "pergunta_reescrita": saida["pergunta_reescrita"], "tipo": saida["tipo"],
        "filtros": saida["filtros_aplicados"], "n_no_recorte": saida["n_no_recorte"], "status": saida["status"],
        "selecionados": [{"review_id": e["review_id"], "score_rerank": e["score_rerank"],
                          "score_portao": round(estado.get("portao", {}).get(e["review_id"], 0.0), 3)}
                         for e in saida["evidencias"]],
        "portao_top1": round(estado.get("portao_top1", 0.0), 3),
        "suporte_lexical": saida.get("suporte_lexical"),
        "citados": sorted({rid for i in saida["insights"] for rid in i["review_ids"]}),
        "validacao": estado.get("validacao"), "limiar_existencia": LIMIAR_EXISTENCIA,
        "prompts": _versao_prompts(),
        "contagem": saida.get("contagem"),
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
        (lambda e: e["comparacao_nao_suportada"],
         _recusa("evidencia_fraca", "A pergunta compara dois recortes, mas esta versão analisa um recorte por vez. "
                                    "Não inferi qual grupo reclama mais sem calcular ambos sob o mesmo critério.")),
        (lambda e: e["n_no_recorte"] == 0,
         _recusa("filtro_sem_resultado", "Não há nenhuma avaliação com comentário no recorte pedido "
                                         "(confira os filtros aplicados).")),
        (lambda e: e["analise"].tipo == "agregada", RunnableLambda(_agregar)),
        RunnableLambda(_buscar) | RunnableLambda(_gerar),
    )


@lru_cache
def chain():
    return montar_chain()


def responder(pergunta: str, filtros: dict | None = None) -> dict:
    """Filtros manuais são validados e sobrescrevem apenas os campos informados."""
    manuais = FiltroReviews.model_validate(filtros).como_dict() if filtros else None
    return chain().invoke({"pergunta": pergunta, "filtros_manuais": manuais})


if __name__ == "__main__":
    r = responder(" ".join(sys.argv[1:]) or "O que os clientes relatam sobre atraso na entrega?")
    print(f"\n[{r['status']}] {r['resposta']}\n")
    for i in r["insights"]:
        print(f"  - {i['afirmacao']}  {i['review_ids']}")
    print(f"\nfiltros: {r['filtros_aplicados']} | busca: \"{r['pergunta_reescrita']}\" | tempos: {r['tempos']}")
    print("\nevidências:")
    for e in r["evidencias"]:
        score = "contagem" if e["score_rerank"] is None else f"{e['score_rerank']:+.2f}"
        print(f"  [{score}] nota {e['nota']} | {e['categoria']} | {e['texto'][:90]!r}")
    if "--json" in sys.argv:
        print(json.dumps(r, ensure_ascii=False, indent=1))
