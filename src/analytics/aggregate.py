"""Responde perguntas de frequência contando toda a base textual do recorte.

"Quais são os principais problemas?" pede frequência, e a busca não mede isso:
ela devolve as avaliações mais parecidas com a pergunta, não as mais comuns. Aqui
os números saem de uma contagem com pandas sobre todas as avaliações indexáveis
do recorte, usando os temas de src/analytics/themes.py. Avaliações sem comentário
não têm conteúdo que permita atribuir tema. O LLM só redige, com os números já
calculados.
"""
import re
from dataclasses import dataclass, field

import pandas as pd
from unidecode import unidecode

from src.config import get_settings
from src.retrieval.filters import mascara

MIN_AVALIACOES = 30  # abaixo disso o recorte é pequeno demais para falar de frequência
PALAVRAS_GENERICAS = {
    "avaliacao", "avaliacoes", "cliente", "clientes", "comentario", "comentarios",
    "compra", "compras",
    "elogio", "elogios", "experiencia", "experiencias", "frequencia", "frequente",
    "mais", "padrao", "padroes", "principal", "principais", "problema", "problemas",
    "pedido", "pedidos", "reclamacao", "reclamacoes", "relato", "relatos", "tema", "temas",
}
SINONIMOS_TOKEN = {
    "atrasada": "atraso",
    "atrasado": "atraso",
    "devolver": "devolucao",
    "devolvida": "devolucao",
    "devolvido": "devolucao",
    "fatura": "fiscal",
    "trocar": "troca",
}
SINAIS_MACRO = {
    "entrega": {
        "atraso", "chegou", "demora", "entrega", "entregue", "frete", "receber",
        "recebido", "recebimento",
    },
    "produto": {
        "anuncio", "defeito", "diferente", "errado", "falsificado", "produto",
        "qualidade", "quebrado",
    },
    "atendimento": {"atendimento", "contato", "resposta", "suporte"},
    "pagamento e nota fiscal": {"boleto", "cartao", "fiscal", "nota", "pagamento", "pix"},
    "troca e devolução": {"devolucao", "estorno", "reembolso", "troca"},
    "embalagem": {"embalado", "embalagem", "pacote"},
}
MACROS_GENERICOS = {"produto", "compra em geral"}


@dataclass
class Contagem:
    polaridade: str                     # "nota_baixa", "nota_alta" ou "todas"
    total_no_recorte: int               # avaliações com texto no recorte (denominador)
    total_com_tema: int                 # as que entraram na contagem (nota 3 fica fora)
    macros: list[dict] = field(default_factory=list)
    temas: list[dict] = field(default_factory=list)
    assunto_nao_modelado: str | None = None


class Agregador:
    def __init__(self):
        s = get_settings()
        self.docs = pd.read_parquet(s.data_processed / "documents.parquet")
        self.temas = pd.read_parquet(s.data_processed / "temas.parquet")
        self.textos = self.docs.set_index("review_id").texto_original

    def polaridade(self, pergunta: str, filtros: dict) -> str:
        """Elogio, problema ou ambos, conforme filtro e termos explícitos da pergunta."""
        if (filtros.get("review_score_min") or 0) >= 4:
            return "nota_alta"
        if (filtros.get("review_score_max") or 5) <= 2:
            return "nota_baixa"
        p = pergunta.lower()
        elogio = sum(t in p for t in ("elogi", "positiv", "satisfeit", "gost", "mais valoriza", "bom", "boa"))
        problema = sum(t in p for t in (
            "problema", "reclama", "insatisfeit", "negativ", "defeit", "ruim", "pior",
            "não chegou", "nao chegou", "não entregue", "nao entregue",
            "quebrad", "errad", "falta", "troca", "devolu", "reembolso",
        ))
        # Em "temas de pedidos entregues com atraso", atraso é o recorte, não
        # necessariamente uma solicitação para olhar apenas notas negativas.
        if not filtros.get("entregue_com_atraso"):
            problema += sum(t in p for t in ("atras", "demora"))
        if elogio > problema:
            return "nota_alta"
        if problema > elogio:
            return "nota_baixa"
        return "todas"

    def contar(self, pergunta: str, filtros: dict, temas_alvo: list[str] | None = None, top: int = 6) -> Contagem:
        pol = self.polaridade(pergunta, filtros)
        recorte = self.docs[mascara(self.docs, filtros)]
        t = self.temas[self.temas.review_id.isin(set(recorte.review_id))]
        if pol != "todas":
            t = t[t.grupo == pol]
        # Uma pergunta direcionada só pode ser contada quando o catálogo realmente
        # modela o assunto. Cair silenciosamente na contagem global responderia outra
        # pergunta com números verdadeiros, mas irrelevantes.
        alvo, nao_modelado = self._selecionar_alvo(self.temas, temas_alvo)
        c = Contagem(pol, len(recorte), len(t))
        if nao_modelado:
            c.total_com_tema = 0
            c.assunto_nao_modelado = nao_modelado
            return c
        if not len(t):
            return c
        if alvo:
            selecionado = pd.Series(False, index=t.index)
            for coluna, valores in alvo.items():
                selecionado |= t[coluna].isin(valores)
            t = t[selecionado]

        c.macros = [{"macro": m, "n": int(n), "pct": round(n / len(t) * 100, 1)}
                    for m, n in t.groupby("macro").size().sort_values(ascending=False).items()]
        por_tema = t.groupby(["nome_tema", "macro"]).size().sort_values(ascending=False).head(top)
        c.temas = [{"tema": nome, "macro": macro, "n": int(n), "pct": round(n / len(t) * 100, 1)}
                   for (nome, macro), n in por_tema.items()]
        c.total_com_tema = len(t)
        return c

    @staticmethod
    def _tokens(texto: str) -> set[str]:
        palavras = re.findall(r"[a-z0-9]+", unidecode(texto.lower()))
        normalizadas = set()
        for p in palavras:
            if p in {"a", "as", "com", "da", "das", "de", "do", "dos", "e", "em", "na", "nas", "no", "nos", "o", "os"}:
                continue
            if p.endswith("oes"):
                p = p[:-3] + "ao"
            elif p.endswith("ais"):
                p = p[:-3] + "al"
            elif len(p) > 4 and p.endswith("s"):
                p = p[:-1]
            p = SINONIMOS_TOKEN.get(p, p)
            normalizadas.add(p)
        return normalizadas

    @classmethod
    def _selecionar_alvo(
        cls, t: pd.DataFrame, temas_alvo: list[str] | None
    ) -> tuple[dict[str, set[str]] | None, str | None]:
        """Traduz assuntos para macros/temas existentes.

        Retorna ({coluna: valores}, None) quando encontrou, (None, assunto) quando
        o assunto não está modelado e (None, None) em pergunta geral.
        """
        alvos = []
        for texto in temas_alvo or []:
            tokens = cls._tokens(texto) - PALAVRAS_GENERICAS
            if tokens:
                alvos.append((texto, tokens))
        if not alvos:
            return None, None

        macros_disponiveis = set(t.macro.dropna().unique())
        temas = {
            nome: cls._tokens(nome) - PALAVRAS_GENERICAS
            for nome in t.nome_tema.dropna().unique()
        }
        macros_encontradas: set[str] = set()
        temas_encontrados: set[str] = set()
        sem_correspondencia = []

        for original, tokens in alvos:
            tokens_originais = tokens
            tokens = tokens - PALAVRAS_GENERICAS or tokens_originais
            # Um assunto específico ("nota fiscal", "produto errado") usa o tema
            # quando há correspondência controlada. Assuntos amplos usam a taxonomia
            # explícita de macro-temas, evitando confundir "troca de produto" com
            # qualquer avaliação de produto.
            candidatos_tema = {
                nome for nome, ts in temas.items()
                if ts and (tokens <= ts or ts <= tokens)
            }
            if candidatos_tema and (len(tokens) > 1 or len(candidatos_tema) == 1):
                temas_encontrados.update(candidatos_tema)
                continue

            candidatos_macro = {
                macro for macro, sinais in SINAIS_MACRO.items()
                if tokens_originais & sinais
            }
            especificos = candidatos_macro - MACROS_GENERICOS
            if especificos:
                candidatos_macro = especificos
            candidatos_macro &= macros_disponiveis
            if candidatos_macro:
                macros_encontradas.update(candidatos_macro)
            else:
                sem_correspondencia.append(original)

        if sem_correspondencia:
            return None, ", ".join(sem_correspondencia)
        selecao = {}
        if temas_encontrados:
            selecao["nome_tema"] = temas_encontrados
        if macros_encontradas:
            selecao["macro"] = macros_encontradas
        if selecao:
            return selecao, None
        return None, None

    def representativas(
        self,
        tema: str,
        filtros: dict,
        polaridade: str = "todas",
        n: int = 3,
        candidatos: int = 10,
    ) -> list[str]:
        """Avaliações que melhor representam o tema, dentro do recorte.

        Um tema pode juntar vários grupos do KMeans, e a distância ao centro não é
        comparável entre eles; além disso o nome dado a um grupo pode não descrever
        bem todas as suas avaliações. Por isso pega as mais próximas do centro de
        cada grupo e reordena pelo re-ranker contra o nome do tema, o que descarta
        exemplos que não falam do assunto.
        """
        from src.retrieval.rerank import reordenar

        recorte = set(self.docs.review_id[mascara(self.docs, filtros)])
        t = self.temas[(self.temas.nome_tema == tema) & self.temas.review_id.isin(recorte)]
        if polaridade != "todas":
            t = t[t.grupo == polaridade]
        if not len(t):
            return []
        proximas = t.sort_values("dist_centro").groupby("tema_id").head(candidatos)
        pares = [(rid, self.textos[rid]) for rid in proximas.review_id if rid in self.textos.index]
        return [rid for rid, _ in reordenar(tema, pares)[:n]]

    def contexto(self, c: Contagem, por_tema: dict[str, list[str]], por_id: pd.DataFrame) -> str:
        """Números e exemplos num bloco de texto para o LLM redigir a resposta."""
        from src.generation.context import montar_contexto

        rotulos = {
            "nota_baixa": "avaliações com nota 1 ou 2",
            "nota_alta": "avaliações com nota 4 ou 5",
            "todas": "avaliações com nota 1, 2, 4 ou 5",
        }
        rotulo = rotulos[c.polaridade]
        linhas = [
            f"<contagem base=\"{c.total_com_tema} {rotulo}, "
            f"entre {c.total_no_recorte} avaliações com comentário no recorte\">"
        ]
        linhas += [f"  macro-tema {m['macro']}: {m['n']} ({m['pct']}%)" for m in c.macros]
        for tema in c.temas:
            linhas.append(f"  tema {tema['tema']} ({tema['macro']}): {tema['n']} ({tema['pct']}%)")
        linhas.append("</contagem>")
        for tema, ids in por_tema.items():
            if ids:
                linhas.append(f"<exemplos tema=\"{tema}\">\n{montar_contexto(ids, por_id)}\n</exemplos>")
        return "\n".join(linhas)
