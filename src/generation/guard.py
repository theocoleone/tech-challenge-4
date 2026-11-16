"""Validação da resposta antes de devolvê-la ao usuário.

1. Citações: todo review_id citado precisa estar entre as avaliações enviadas ao
   modelo. Insight com citação inventada é removido.
2. Linguagem de frequência: a resposta de uma pergunta específica vem de uma
   amostra recuperada por busca, então não pode dizer "a maioria" nem dar
   percentuais. Quando isso aparece, a resposta é refeita uma vez com o aviso.
3. IDs no texto: os códigos das avaliações vão nos insights, não na resposta.
4. Números das respostas agregadas: todo número citado tem de vir da contagem.
"""
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from unidecode import unidecode

FREQUENCIA = re.compile(
    r"\b(a maioria|maior parte|grande parte|quase todos|todos os clientes|"
    r"a minoria|frequentemente|na maioria|predomin\w*)\b|\d+\s?%", re.IGNORECASE)
ID_AVALIACAO = re.compile(r"\(?\b(?:ID:?\s*)?[0-9a-f]{32}\b\)?,?", re.IGNORECASE)


@dataclass
class Validacao:
    insights: list[dict]
    citacoes_invalidas: int
    linguagem_de_frequencia: list[str]


def validar(resposta: str, insights: list[dict], ids_contexto: set[str]) -> Validacao:
    validos, invalidas = [], 0
    for ins in insights:
        ids = [rid for rid in ins.get("review_ids", []) if rid in ids_contexto]
        invalidas += len(ins.get("review_ids", [])) - len(ids)
        if ids:
            validos.append({
                **ins,
                "afirmacao": limpar_ids(ins.get("afirmacao", "")),
                "review_ids": ids,
            })
    achados = [m.group(0) for m in FREQUENCIA.finditer(resposta)]
    return Validacao(validos, invalidas, achados)


def limpar_ids(texto: str) -> str:
    sem_ids = ID_AVALIACAO.sub("", texto)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", sem_ids)).strip()


NUMERO = re.compile(
    r"(?<![\w])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(\s*%)?(?![\w])"
)


def _numero_canonico(valor) -> str:
    texto = str(valor).strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", texto):
        texto = texto.replace(".", "").replace(",", ".")
    elif "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    try:
        numero = Decimal(texto)
    except InvalidOperation:
        return texto
    return format(numero.normalize(), "f")


def _distancia(inicio: int, fim: int, outro_inicio: int, outro_fim: int) -> int:
    if outro_fim <= inicio:
        return inicio - outro_fim
    if outro_inicio >= fim:
        return outro_inicio - fim
    return 0


def _rotulos_numericos(texto: str, contagem) -> list[dict]:
    """Localiza os nomes de temas/macros e associa a eles seus números permitidos."""
    normalizado = unidecode(texto.lower())
    itens = list(contagem.macros) + list(contagem.temas)
    tokens_por_item = [
        set(re.findall(r"[a-z0-9]+", unidecode(str(item.get("tema") or item.get("macro") or "").lower())))
        for item in itens
    ]
    frequencia_token = {
        token: sum(token in tokens for tokens in tokens_por_item)
        for tokens in tokens_por_item
        for token in tokens
    }
    aliases_ignorados = {
        "avaliacao", "compra", "entrega", "geral", "nota", "problema", "produto",
        "qualidade",
    }
    ocorrencias = []
    for item, tokens in zip(itens, tokens_por_item):
        rotulo = unidecode(str(item.get("tema") or item.get("macro") or "").lower())
        if not rotulo:
            continue
        aliases = {rotulo} | {
            token for token in tokens
            if len(token) >= 5 and token not in aliases_ignorados and frequencia_token[token] == 1
        }
        for alias in aliases:
            for achado in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", normalizado):
                ocorrencias.append({
                    "inicio": achado.start(),
                    "fim": achado.end(),
                    "tamanho": len(alias),
                    "contagens": {_numero_canonico(item["n"])},
                    "percentuais": {
                        _numero_canonico(item["pct"]),
                        _numero_canonico(round(item["pct"])),
                    },
                })
    return ocorrencias


def _permitido_pelo_contexto(
    texto: str,
    achado: re.Match,
    canonico: str,
    contagem,
    filtros: dict,
) -> bool:
    """Permite notas/períodos/valores somente quando a unidade aparece ao redor."""
    inicio, fim = achado.span()
    janela = unidecode(texto[max(0, inicio - 35): min(len(texto), fim + 35)].lower())
    notas = {
        "nota_baixa": {1, 2},
        "nota_alta": {4, 5},
        "todas": {1, 2, 4, 5},
    }[contagem.polaridade]
    if canonico in {_numero_canonico(n) for n in notas} and "nota" in janela:
        return True

    for chave, valor in filtros.items():
        if isinstance(valor, bool):
            continue
        permitido = {_numero_canonico(valor)}
        if chave.startswith("ano_mes_compra") and isinstance(valor, (int, float)):
            permitido.add(str(int(valor) // 100))
        if canonico not in permitido:
            continue
        if chave.startswith("valor_") and any(x in janela for x in ("r$", "real", "valor", "acima de", "abaixo de")):
            return True
        if chave.startswith("ano_mes_compra") and any(x in janela for x in ("ano", "compra", "mes", "periodo")):
            return True
        if chave.startswith("review_score") and "nota" in janela:
            return True
    return False


def conferir_numeros(resposta: str, contagem, filtros: dict | None = None) -> list[str]:
    """Números da resposta agregada que não estão na contagem (o LLM não pode criar nenhum).

    Distingue contagens de percentuais, normaliza a notação brasileira de milhar e
    só permite notas/períodos que realmente apareçam no contexto ou nos filtros.
    """
    bases = {
        _numero_canonico(contagem.total_no_recorte),
        _numero_canonico(contagem.total_com_tema),
    }
    rotulos = _rotulos_numericos(resposta, contagem)
    invalidos = []
    for achado in NUMERO.finditer(resposta):
        original, unidade = achado.group(1), achado.group(2)
        canonico = _numero_canonico(original)
        if not unidade and (
            canonico in bases
            or _permitido_pelo_contexto(resposta, achado, canonico, contagem, filtros or {})
        ):
            continue

        anteriores = []
        posteriores = []
        for rotulo in rotulos:
            distancia = _distancia(achado.start(), achado.end(), rotulo["inicio"], rotulo["fim"])
            if distancia > 120:
                continue
            destino = anteriores if rotulo["fim"] <= achado.start() else posteriores
            destino.append((distancia, -rotulo["tamanho"], rotulo))
        # O prompt pede "tema: número". Dar prioridade ao rótulo anterior evita
        # atribuir o percentual ao próximo tema só porque ele começa logo depois.
        proximos = anteriores or posteriores
        if proximos:
            melhor_chave = min((d, tamanho) for d, tamanho, _ in proximos)
            melhores = [r for d, tamanho, r in proximos if (d, tamanho) == melhor_chave]
            permitidos = set().union(*(
                r["percentuais"] if unidade else r["contagens"] for r in melhores
            ))
        else:
            permitidos = set()
        if canonico not in permitidos:
            invalidos.append(original + ("%" if unidade else ""))
    return invalidos
