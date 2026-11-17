"""Suporte lexical determinístico para decidir se um assunto existe no corpus.

Este módulo não substitui a recuperação semântica. Ele fornece um sinal
independente e auditável para assuntos explícitos: quantas avaliações, dentro
do recorte de metadados, contêm os termos-alvo com limites de palavra.

Os itens de ``temas_alvo`` são tratados como alternativas. Dentro de um item
com mais de uma palavra, todas precisam aparecer na mesma avaliação. Quando o
LLM não produz temas, a pergunta é usada como fallback após retirar palavras
funcionais e termos genéricos de formulação.
"""
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Literal
import unicodedata

import numpy as np
import pandas as pd

from src.retrieval.filters import mascara

FaixaSuporte = Literal["sem_evidencia", "evidencia_fraca", "evidencia_suficiente"]
OrigemTermos = Literal["temas_alvo", "pergunta", "nenhuma"]

MAX_EVIDENCIAS_FRACAS = 3

# Palavras funcionais e de formulação, que não definem o assunto procurado.
_GENERICO = set(
    """
    a ao aos aquela aquelas aquele aqueles aquilo as ate avalia avaliam avaliar
    avaliacao avaliacoes cliente clientes acha acham achar como comprado
    comprados com comenta comentam comentario comentarios da das de dela delas
    dele deles diz dizem dizer do dos e ela
    elas ele eles em entre era essa essas esse esses esta estao avaliaram existe
    existem faz fazem foi foram geral ha mais me mesmo minha muito na nas
    no nos nossa nossas nosso nossos o os ou para pela pelas pelo pelos por
    porque principais problema problemas quais que
    reclamacao reclamacoes reclamam reclamar reclamaram relata relatam relatar
    relatada relatadas relatado relatados relato relatos sao se seu seus sobre
    sua suas tema temas tentar um uma umas uns vendido vendida vendidos
    vendidas
    """.split()
)

# Termos que qualificam um núcleo mais específico. "Atendimento WhatsApp"
# procura WhatsApp; "juros no parcelamento" procura juros. Se vierem sozinhos
# ("problemas na entrega"), continuam sendo um assunto válido.
_MODIFICADORES_DOMINIO = set(
    """
    aplicativo aplicativos atendimento atendimentos canal canais cobranca
    cobrancas compra compras entrega entregas frete fretes loja lojas nota notas
    pagamento pagamentos parcelamento parcelamentos pedido pedidos produto
    produtos programa programas site sites valor valores
    """.split()
)

_ALTERNATIVA = re.compile(r"\s+\bou\b\s+|[,;/|]+", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")


def normalizar_lexico(texto: object) -> str:
    """Normaliza caixa, acentos e separadores sem alterar limites de palavra."""
    if texto is None or pd.isna(texto):
        return ""
    decomposto = unicodedata.normalize("NFKD", str(texto).casefold())
    sem_acentos = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(_TOKEN.findall(sem_acentos))


def _variantes(token: str) -> tuple[str, ...]:
    """Produz variações simples de singular/plural sem fazer stemming."""
    formas = {token}
    if len(token) < 4:
        return tuple(formas)

    if token.endswith("oes"):
        formas.add(token[:-3] + "ao")
    elif token.endswith("ais"):
        formas.add(token[:-3] + "al")
    elif token.endswith("eis"):
        formas.add(token[:-3] + "el")
    elif token.endswith("ns"):
        formas.add(token[:-2] + "m")
    elif token.endswith("s") and not token.endswith(("ss", "us")):
        formas.add(token[:-1])
    elif token.endswith("ao"):
        formas.add(token[:-2] + "oes")
    elif token.endswith("al"):
        formas.add(token[:-2] + "ais")
    elif token.endswith("el"):
        formas.add(token[:-2] + "eis")
    elif token.endswith("m"):
        formas.add(token[:-1] + "ns")
    else:
        formas.add(token + "s")
    return tuple(sorted(formas))


def _padrao_token(token: str) -> str:
    variantes = "|".join(re.escape(v) for v in _variantes(token))
    # O texto já foi normalizado para [a-z0-9 ]: os lookarounds deixam
    # explícito que "senha" não pode casar com "resenha".
    return rf"(?<![a-z0-9])(?:{variantes})(?![a-z0-9])"


def _priorizar_nucleo(
    grupos: Iterable[tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    """Remove modificadores quando existe ao menos um núcleo mais específico."""
    brutos: list[tuple[str, ...]] = []
    nucleos: list[tuple[str, ...]] = []
    for grupo in grupos:
        limpo = tuple(token for token in grupo if token not in _GENERICO)
        if limpo and limpo not in brutos:
            brutos.append(limpo)
        nucleo = tuple(token for token in limpo if token not in _MODIFICADORES_DOMINIO)
        if nucleo and nucleo not in nucleos:
            nucleos.append(nucleo)
    return tuple(nucleos or brutos)


def _grupos_dos_temas(temas_alvo: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    grupos: list[tuple[str, ...]] = []
    for tema in temas_alvo:
        for alternativa in _ALTERNATIVA.split(str(tema)):
            tokens = tuple(_TOKEN.findall(normalizar_lexico(alternativa)))
            if tokens and tokens not in grupos:
                grupos.append(tokens)
    return _priorizar_nucleo(grupos)


def _grupos_da_pergunta(pergunta: str) -> tuple[tuple[str, ...], ...]:
    grupos: list[tuple[str, ...]] = []
    for alternativa in _ALTERNATIVA.split(pergunta):
        tokens = tuple(
            token for token in _TOKEN.findall(normalizar_lexico(alternativa))
            if len(token) >= 3 and not token.isdigit()
        )
        if tokens and tokens not in grupos:
            grupos.append(tokens)
    return _priorizar_nucleo(grupos)


def classificar_quantidade(quantidade: int) -> FaixaSuporte:
    """Converte a contagem no critério 0, 1–3 e 4 ou mais."""
    if quantidade < 0:
        raise ValueError("quantidade não pode ser negativa")
    if quantidade == 0:
        return "sem_evidencia"
    if quantidade <= MAX_EVIDENCIAS_FRACAS:
        return "evidencia_fraca"
    return "evidencia_suficiente"


@dataclass(frozen=True)
class SuporteLexical:
    aplicavel: bool
    faixa: FaixaSuporte | None
    quantidade: int
    review_ids: tuple[str, ...]
    grupos_de_termos: tuple[tuple[str, ...], ...]
    origem_termos: OrigemTermos
    total_no_recorte: int


class VerificadorEvidenciaLexical:
    """Conta evidências exatas em um DataFrame ou em documents.parquet."""

    def __init__(self, documentos: pd.DataFrame):
        ausentes = {"review_id"} - set(documentos.columns)
        if ausentes:
            raise ValueError(f"colunas obrigatórias ausentes: {sorted(ausentes)}")
        coluna_texto = (
            "texto_normalizado" if "texto_normalizado" in documentos.columns
            else "texto_original" if "texto_original" in documentos.columns
            else None
        )
        if coluna_texto is None:
            raise ValueError("documents precisa ter texto_normalizado ou texto_original")
        if documentos.review_id.isna().any():
            raise ValueError("review_id não pode ser nulo")

        self.documentos = documentos
        self._textos = documentos[coluna_texto].map(normalizar_lexico)

    @classmethod
    def de_parquet(cls, caminho: str | Path) -> "VerificadorEvidenciaLexical":
        return cls(pd.read_parquet(Path(caminho)))

    def avaliar(
        self,
        pergunta: str,
        temas_alvo: str | Iterable[str] | None = None,
        filtros: dict | None = None,
    ) -> SuporteLexical:
        temas_recebidos = (temas_alvo,) if isinstance(temas_alvo, str) else (temas_alvo or ())
        temas = tuple(t for t in temas_recebidos if normalizar_lexico(t))
        if temas:
            grupos = _grupos_dos_temas(temas)
            origem: OrigemTermos = "temas_alvo"
        else:
            grupos = _grupos_da_pergunta(pergunta)
            origem = "pergunta" if grupos else "nenhuma"

        no_recorte = mascara(self.documentos, filtros)
        total_no_recorte = int(no_recorte.sum())
        if not grupos:
            return SuporteLexical(
                aplicavel=False,
                faixa=None,
                quantidade=0,
                review_ids=(),
                grupos_de_termos=(),
                origem_termos=origem,
                total_no_recorte=total_no_recorte,
            )

        encontrados = np.zeros(len(self.documentos), dtype=bool)
        for grupo in grupos:
            contem_grupo = np.ones(len(self.documentos), dtype=bool)
            for token in grupo:
                contem_grupo &= self._textos.str.contains(
                    _padrao_token(token), regex=True, na=False
                ).to_numpy()
            encontrados |= contem_grupo
        encontrados &= no_recorte

        # documents.parquet tem um documento por review_id. O dict também
        # preserva uma saída correta e determinística para DataFrames sintéticos
        # que contenham IDs repetidos.
        ids = tuple(dict.fromkeys(self.documentos.loc[encontrados, "review_id"].astype(str)))
        quantidade = len(ids)
        return SuporteLexical(
            aplicavel=True,
            faixa=classificar_quantidade(quantidade),
            quantidade=quantidade,
            review_ids=ids,
            grupos_de_termos=grupos,
            origem_termos=origem,
            total_no_recorte=total_no_recorte,
        )
