"""Monta a base de conhecimento do RAG a partir dos CSVs da Olist.

Saídas em data/processed/:
- reviews.parquet: TODAS as avaliações (com e sem texto) e seus metadados.
  É a base das contagens (nota, período, categoria), inclusive das perguntas
  agregadas, e permite declarar o viés "quem comenta".
- documents.parquet: só as avaliações com texto informativo. 1 avaliação =
  1 documento, sem chunking (as avaliações têm no máximo ~45 palavras).

Uso: python -m src.data.build_documents
"""
import json
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.data.clean import contar_palavras, informativo, juntar_titulo_mensagem, normalizar
from src.data.load_olist import load_olist
from src.data.pii import mascarar_pii


def _yyyymmdd(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")


def _metadados_por_pedido(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uma linha por order_id. Agrega os itens ANTES do join para não multiplicar avaliações."""
    itens = t["items"].merge(t["products"][["product_id", "product_category_name"]], on="product_id", how="left")
    itens = itens.merge(t["categorias"], on="product_category_name", how="left")
    itens["product_category_name"] = itens["product_category_name"].fillna("sem_categoria")
    itens["product_category_name_english"] = itens["product_category_name_english"].fillna("no_category")
    itens["valor_item"] = itens["price"] + itens["freight_value"]

    principal = (
        itens.sort_values("valor_item", ascending=False)  # categoria do item de maior valor
        .drop_duplicates("order_id")[["order_id", "product_category_name", "product_category_name_english"]]
        .rename(columns={"product_category_name": "categoria_pt", "product_category_name_english": "categoria_en"})
    )
    agregados = itens.groupby("order_id").agg(
        valor_total=("valor_item", "sum"),
        n_itens=("order_item_id", "count"),
        n_categorias=("product_category_name", "nunique"),
    )

    pedidos = t["orders"].merge(t["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
    pedidos = pedidos.merge(principal, on="order_id", how="left").merge(agregados, on="order_id", how="left")
    pedidos["categoria_pt"] = pedidos["categoria_pt"].fillna("sem_categoria")
    pedidos["categoria_en"] = pedidos["categoria_en"].fillna("no_category")
    pedidos["n_itens"] = pedidos["n_itens"].fillna(0).astype(int)
    pedidos["multi_categoria"] = pedidos["n_categorias"].fillna(0) > 1

    entregue = pedidos["order_delivered_customer_date"]
    pedidos["entregue"] = entregue.notna()
    atraso = (entregue.dt.normalize() - pedidos["order_estimated_delivery_date"].dt.normalize()).dt.days
    pedidos["dias_atraso"] = atraso.astype("Int64")  # nulo quando não entregue; não imputar
    pedidos["entregue_com_atraso"] = (atraso > 0).fillna(False)
    pedidos["ano_mes_compra"] = pd.to_numeric(
        pedidos["order_purchase_timestamp"].dt.strftime("%Y%m"), errors="coerce"
    ).astype("Int64")
    return pedidos.rename(columns={"customer_state": "uf_cliente"})


def _uma_linha_por_review(reviews: pd.DataFrame, pedidos: pd.DataFrame) -> pd.DataFrame:
    """review_id em vários pedidos vira uma linha só, com os metadados do pedido mais recente."""
    r = reviews.merge(pedidos, on="order_id", how="left")
    order_ids = r.groupby("review_id")["order_id"].agg(lambda s: ",".join(sorted(set(s))))
    n_pedidos = r.groupby("review_id")["order_id"].nunique()
    r = r.sort_values("order_purchase_timestamp", ascending=False).drop_duplicates("review_id")
    r["order_ids"] = r["review_id"].map(order_ids)
    r["multi_pedido"] = r["review_id"].map(n_pedidos) > 1
    return r


def build(raw_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    s = get_settings()
    out_dir = Path(out_dir or s.data_processed)
    t = load_olist(raw_dir)
    stats = {"avaliacoes_brutas": len(t["reviews"])}

    r = _uma_linha_por_review(t["reviews"], _metadados_por_pedido(t))
    stats["avaliacoes_unicas"] = len(r)
    stats["review_id_em_varios_pedidos"] = int(r["multi_pedido"].sum())
    stats["pedidos_com_varias_avaliacoes"] = int((t["reviews"].groupby("order_id")["review_id"].nunique() > 1).sum())

    r["tem_titulo"] = r["review_comment_title"].notna()
    r["tem_mensagem"] = r["review_comment_message"].notna()
    bruto = [juntar_titulo_mensagem(a, b) for a, b in zip(r["review_comment_title"], r["review_comment_message"])]
    mascarado = [mascarar_pii(x) for x in bruto]
    r["texto_original"] = [x for x, _ in mascarado]
    pii_total: dict[str, int] = {}
    for _, c in mascarado:
        for k, v in c.items():
            pii_total[k] = pii_total.get(k, 0) + v
    r["texto_normalizado"] = r["texto_original"].map(normalizar)
    r["n_palavras"] = r["texto_normalizado"].map(contar_palavras)
    r["tem_texto"] = r["texto_original"] != ""
    r["indexavel"] = r["texto_normalizado"].map(informativo)

    r["data_avaliacao"] = _yyyymmdd(r["review_answer_timestamp"])
    r["data_envio_pesquisa"] = _yyyymmdd(r["review_creation_date"])

    colunas = [
        "review_id", "order_ids", "review_score", "texto_original", "texto_normalizado",
        "tem_texto", "tem_titulo", "tem_mensagem", "indexavel", "n_palavras",
        "data_avaliacao", "data_envio_pesquisa", "ano_mes_compra", "order_status",
        "valor_total", "multi_pedido", "entregue", "dias_atraso", "entregue_com_atraso",
        "categoria_pt", "categoria_en", "multi_categoria", "n_itens", "uf_cliente",
    ]
    r = r[colunas].reset_index(drop=True)
    docs = r[r["indexavel"]].reset_index(drop=True)

    stats.update(
        sem_texto=int((~r["tem_texto"]).sum()),
        com_texto=int(r["tem_texto"].sum()),
        so_titulo=int((r["tem_titulo"] & ~r["tem_mensagem"]).sum()),
        texto_nao_informativo=int((r["tem_texto"] & ~r["indexavel"]).sum()),
        documentos=len(docs),
        pii_mascarada=pii_total,
        n_palavras_p50_p99_max=[int(docs["n_palavras"].quantile(q)) for q in (0.5, 0.99)] + [int(docs["n_palavras"].max())],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out_dir / "reviews.parquet", index=False)
    docs.to_parquet(out_dir / "documents.parquet", index=False)
    (out_dir / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
