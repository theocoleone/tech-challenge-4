# Dados

Base: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) (Kaggle). Os CSVs não são versionados.

## Como baixar

1. Na página do Kaggle, clique em **Download** (ou use a CLI: `kaggle datasets download -d olistbr/brazilian-ecommerce`).
2. Descompacte em `data/raw/`:

```bash
mkdir -p data/raw && unzip ~/Downloads/archive.zip -d data/raw
```

## Arquivos usados

| Arquivo | Para quê |
|---|---|
| `olist_order_reviews_dataset.csv` | Avaliações: nota, título, comentário, datas (o texto do RAG) |
| `olist_orders_dataset.csv` | Status do pedido e datas de compra, entrega estimada e real (atraso) |
| `olist_order_items_dataset.csv` | Itens, preço e frete do pedido |
| `olist_products_dataset.csv` | Categoria do produto (em português) |
| `product_category_name_translation.csv` | Tradução da categoria para inglês |
| `olist_customers_dataset.csv` | Estado/cidade do cliente |

`data/processed/` guarda os arquivos gerados pelo pipeline (ex.: `documents.parquet`).
