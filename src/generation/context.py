"""Montagem do contexto enviado ao LLM: cada avaliação vira uma tag <review>.

Os metadados entram como atributos para o modelo poder citar nota, data e
categoria, e o id é o que ele usa nas citações. O texto do cliente fica dentro
da tag, separado das instruções do sistema (defesa contra prompt injection).
"""
from xml.sax.saxutils import escape, quoteattr

import pandas as pd


def montar_contexto(review_ids: list[str], por_id: pd.DataFrame) -> str:
    blocos = []
    for rid in review_ids:
        d = por_id.loc[rid]
        atributos = {
            "id": rid, "nota": str(int(d.review_score)), "categoria": d.categoria_pt,
            "mes_compra": str(d.ano_mes_compra), "entregue_com_atraso": "sim" if d.entregue_com_atraso else "não",
        }
        attrs = " ".join(f"{k}={quoteattr(v)}" for k, v in atributos.items())
        blocos.append(f"<review {attrs}>{escape(d.texto_original)}</review>")
    return "<avaliacoes>\n" + "\n".join(blocos) + "\n</avaliacoes>"
