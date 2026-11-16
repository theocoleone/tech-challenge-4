"""Temas das avaliações, calculados uma vez sobre toda a base textual.

Perguntas como "quais são os principais problemas?" pedem frequência. A busca
devolve as avaliações mais parecidas com a pergunta, não uma contagem, então
não responde isso. Aqui as avaliações são agrupadas em temas para poder contar.

- Avaliações de nota baixa (1 e 2) e de nota alta (4 e 5) são agrupadas
  separadamente, com KMeans sobre os vetores do Titan (25 e 20 grupos).
- Cada grupo recebe um nome, uma descrição e um macro-tema de uma lista fixa,
  dados pelo LLM a partir das avaliações mais próximas do centro do grupo.
- Grupos diferentes que recebem o mesmo nome (ou um sinônimo da tabela ALIASES)
  são **unidos** num tema só: o KMeans separa por jeitos de escrever ("atraso na
  entrega" saiu em 9 grupos), e para contar frequência o que importa é o assunto.
- Cada tema também tem um macro-tema (entrega, produto, atendimento...), o que dá
  dois níveis de leitura: o macro para o panorama e o tema para o detalhe.
- A nota 3 fica fora: são avaliações mistas.

Saídas: data/processed/temas.parquet (review_id -> tema) e o catálogo dos temas
em eval/results/temas_catalogo.json (versionado, para revisão dos nomes).

Uso: python -m src.analytics.themes
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from src.config import ROOT, get_llm, get_settings
from src.index.embeddings import MODELOS, identidade_modelo
from src.index.vector_store import IndiceVetorial

MACROS = ["entrega", "produto", "atendimento", "pagamento e nota fiscal", "troca e devolução", "embalagem", "compra em geral"]
GRUPOS = {"nota_baixa": ((1, 2), 25, "nota 1 ou 2"), "nota_alta": ((4, 5), 20, "nota 4 ou 5")}
N_EXEMPLOS = 25


class RotuloTema(BaseModel):
    nome: str = Field(description="nome curto do assunto, de 2 a 5 palavras")
    descricao: str = Field(description="uma frase descrevendo o que as avaliações dizem")
    macro: str = Field(description=f"um destes: {', '.join(MACROS)}")


def _rotular(polaridade: str, exemplos: list[str]) -> RotuloTema:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (ROOT / "src" / "generation" / "prompts" / "rotulo_tema_v1.md").read_text()),
        ("human", "Avaliações:\n{avaliacoes}"),
    ])
    cadeia = prompt | get_llm(estruturado=RotuloTema, papel="geracao")
    r = cadeia.invoke({"polaridade": polaridade, "avaliacoes": "\n".join(f"- {t}" for t in exemplos)})
    if r.macro not in MACROS:
        r.macro = "compra em geral"
    return r


# Sinônimos: chave normalizada -> chave canônica. Dois grupos com a mesma chave já
# são unidos automaticamente; esta tabela cobre os casos em que o LLM usou palavras
# diferentes para o mesmo assunto. O mapeamento é deliberadamente conservador:
# assuntos distintos ("produto quebrado" x "produto errado") ficam separados.
# O nome exibido de cada tema é o do maior grupo que caiu nele.
ALIASES = {
    "demora_entrega": "atraso_entrega",
    "antecipada_entrega": "entrega_rapida",
    "entregue_nao_produto": "nao_produto_recebido",
    "nao_recebi": "nao_produto_recebido",
    "entrega_parcial": "incompleto_pedido",
    "falta_produto": "incompleto_pedido",
    "faltante_produto": "incompleto_pedido",
    "diferente_defeito_ou_produto": "anuncio_diferente_produto",
    "otimo_produto": "produto_qualidade",
    "geral_recomendacao": "avaliacao_positiva",
    "recomendacao": "avaliacao_positiva",
    "geral_satisfacao": "avaliacao_positiva",
    "compra_experiencia_positiva": "avaliacao_positiva",
    "gostei_muito": "avaliacao_positiva",
    "ok_tudo": "avaliacao_positiva",
    "entrega_problema": "entrega_problema",
}


def _chave(nome: str) -> str:
    """Nome do tema normalizado (sem acento, caixa, artigos nem plural), para unir
    grupos que falam do mesmo assunto: "Atraso na entrega" e "atrasos entrega"."""
    import re
    import unicodedata

    t = unicodedata.normalize("NFKD", nome.lower()).encode("ascii", "ignore").decode()
    palavras = [p for p in re.findall(r"[a-z]+", t) if p not in {"na", "no", "de", "do", "da", "com", "em", "a", "o", "os", "as"}]
    return "_".join(sorted(p[:-1] if len(p) > 4 and p.endswith("s") else p for p in palavras))


def construir() -> pd.DataFrame:
    s = get_settings()
    docs = pd.read_parquet(s.data_processed / "documents.parquet")
    vetores = IndiceVetorial.carregar(
        s.index_dir,
        "titan",
        model_id_esperado=identidade_modelo("titan"),
        model_id_legado=MODELOS["titan"],
    )
    linhas, catalogo = [], []
    for grupo, (notas, k, descricao_grupo) in GRUPOS.items():
        parte = docs[docs.review_score.isin(notas)].reset_index(drop=True)
        X = vetores.vetores_de(parte.review_id.tolist())
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
        for c in range(k):
            idx = np.flatnonzero(km.labels_ == c)
            perto = idx[np.argsort(dist[idx])][:N_EXEMPLOS]
            # textos repetidos ("muito bom") não ajudam a nomear: prioriza exemplos distintos
            exemplos = list(dict.fromkeys(parte.texto_original.iloc[perto]))
            rot = _rotular(descricao_grupo, exemplos)
            tema_id = f"{grupo}_{c:02d}"
            catalogo.append({"tema_id": tema_id, "grupo": grupo, "nome": rot.nome, "descricao": rot.descricao,
                             "macro": rot.macro, "n_avaliacoes": int(len(idx)),
                             "exemplos": parte.texto_original.iloc[perto[:5]].tolist(),
                             "centroide_review_ids": parte.review_id.iloc[perto[:10]].tolist()})
            print(f"{tema_id} ({len(idx):>5}) [{rot.macro}] {rot.nome}", flush=True)
        linhas.append(pd.DataFrame({"review_id": parte.review_id, "grupo": grupo,
                                    "tema_id": [f"{grupo}_{c:02d}" for c in km.labels_], "dist_centro": dist}))
    temas = pd.concat(linhas, ignore_index=True)
    cat = pd.DataFrame(catalogo)
    cat["chave"] = cat.nome.map(_chave).map(lambda k: ALIASES.get(k, k))
    cat["tema"] = cat.grupo + ":" + cat.chave
    # nome exibido e macro-tema vêm do maior grupo que caiu no tema
    maior = cat.sort_values("n_avaliacoes", ascending=False).drop_duplicates("tema").set_index("tema")
    cat["nome_tema"] = cat.tema.map(maior.nome)
    cat["macro"] = cat.tema.map(maior.macro)
    temas = temas.merge(cat[["tema_id", "tema", "nome_tema", "macro"]], on="tema_id")
    unidos = cat.groupby("tema").tema_id.count()
    print(f"\n{len(cat)} grupos -> {cat.tema.nunique()} temas "
          f"({(unidos > 1).sum()} unidos, maior junta {unidos.max()} grupos)")
    temas.to_parquet(s.data_processed / "temas.parquet", index=False)
    (ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval" / "results" / "temas_catalogo.json").write_text(
        json.dumps(cat.drop(columns="centroide_review_ids").to_dict("records"), ensure_ascii=False, indent=1))
    return temas


if __name__ == "__main__":
    construir()
