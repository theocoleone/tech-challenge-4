"""Leitura e tipagem dos CSVs da Olist usados pelo TC4."""
from pathlib import Path

import pandas as pd

from src.config import get_settings

ARQUIVOS = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "categorias": "product_category_name_translation.csv",
    "customers": "olist_customers_dataset.csv",
}

DATAS = {
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "orders": [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}


def load_olist(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Lê as 6 tabelas e converte as colunas de data. Falha cedo se faltar arquivo."""
    raw_dir = Path(raw_dir or get_settings().data_raw)
    faltando = [nome for nome in ARQUIVOS.values() if not (raw_dir / nome).exists()]
    if faltando:
        raise FileNotFoundError(f"Faltam em {raw_dir}: {faltando}. Ver data/README.md")

    tabelas = {}
    for chave, nome in ARQUIVOS.items():
        df = pd.read_csv(raw_dir / nome)
        for col in DATAS.get(chave, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        tabelas[chave] = df
    return tabelas
