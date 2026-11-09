"""Regras da base de conhecimento (seção 4.2 do planning), com dados sintéticos."""
import pandas as pd
import pytest

from src.data.build_documents import build
from src.data.clean import contar_palavras, informativo, juntar_titulo_mensagem, normalizar
from src.data.pii import mascarar_pii


@pytest.fixture
def base(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4", "r4", "r5"],
        "order_id": ["o1", "o1", "o2", "o3", "o4", "o5"],
        "review_score": [1, 5, 4, 2, 2, 3],
        "review_comment_title": [None, "Ótimo", None, None, None, None],
        "review_comment_message": ["Não chegou, vc acredita?", None, None, "Atrasou muito", "Atrasou muito", "..."],
        "review_creation_date": ["2018-01-10"] * 6,
        "review_answer_timestamp": ["2018-01-12 10:00:00"] * 6,
    }).to_csv(raw / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "customer_id": ["c1"] * 5,
        "order_status": ["delivered", "delivered", "delivered", "delivered", "canceled"],
        "order_purchase_timestamp": ["2017-12-01", "2017-12-02", "2017-11-01", "2018-01-01", "2017-12-05"],
        "order_delivered_customer_date": ["2017-12-20", "2017-12-10", "2017-11-30", "2018-01-05", None],
        "order_estimated_delivery_date": ["2017-12-15", "2017-12-20", "2017-11-20", "2018-01-10", "2017-12-30"],
    }).to_csv(raw / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": ["o1", "o1", "o2", "o3", "o4", "o5"],
        "order_item_id": [1, 2, 1, 1, 1, 1],
        "product_id": ["pA", "pB", "pA", "pA", "pB", "pX"],
        "price": [10.0, 100.0, 10.0, 10.0, 100.0, 5.0],
        "freight_value": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
    }).to_csv(raw / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame({
        "product_id": ["pA", "pB", "pX"],
        "product_category_name": ["cama_mesa_banho", "informatica_acessorios", None],
    }).to_csv(raw / "olist_products_dataset.csv", index=False)
    pd.DataFrame({
        "product_category_name": ["cama_mesa_banho", "informatica_acessorios"],
        "product_category_name_english": ["bed_bath_table", "computers_accessories"],
    }).to_csv(raw / "product_category_name_translation.csv", index=False)
    pd.DataFrame({"customer_id": ["c1"], "customer_state": ["SP"]}).to_csv(
        raw / "olist_customers_dataset.csv", index=False)

    out = tmp_path / "processed"
    stats = build(raw, out)
    return stats, pd.read_parquet(out / "reviews.parquet"), pd.read_parquet(out / "documents.parquet")


def test_review_em_varios_pedidos_vira_um_documento_com_pedido_mais_recente(base):
    _, reviews, _ = base
    r4 = reviews[reviews.review_id == "r4"]
    assert len(r4) == 1
    r4 = r4.iloc[0]
    assert r4.multi_pedido and r4.order_ids == "o3,o4"
    assert r4.ano_mes_compra == 201801  # o4, o mais recente


def test_pedido_com_varias_avaliacoes_mantem_todas(base):
    stats, reviews, _ = base
    assert set(reviews[reviews.order_ids == "o1"].review_id) == {"r1", "r2"}
    assert stats["pedidos_com_varias_avaliacoes"] == 1


def test_sem_texto_fica_nas_contagens_mas_fora_do_indice(base):
    _, reviews, docs = base
    assert "r3" in set(reviews.review_id)
    assert "r3" not in set(docs.review_id)
    assert not reviews.set_index("review_id").loc["r5", "indexavel"]  # só pontuação


def test_metadados_de_entrega_e_categoria(base):
    _, reviews, _ = base
    r = reviews.set_index("review_id")
    assert r.loc["r1", "dias_atraso"] == 5 and r.loc["r1", "entregue_com_atraso"]
    assert r.loc["r1", "categoria_pt"] == "informatica_acessorios"  # item de maior valor
    assert r.loc["r1", "multi_categoria"]
    assert r.loc["r1", "valor_total"] == pytest.approx(113.0)
    assert not r.loc["r5", "entregue"] and pd.isna(r.loc["r5", "dias_atraso"])  # não imputa
    assert r.loc["r5", "categoria_pt"] == "sem_categoria"


def test_texto_original_preservado_e_normalizado_expande_abreviacao(base):
    _, _, docs = base
    d = docs.set_index("review_id").loc["r1"]
    assert d.texto_original == "Não chegou, vc acredita?"
    assert d.texto_normalizado == "Não chegou , você acredita ?"


def test_normalizar_mantem_negacao_e_acentos_e_e_idempotente():
    t = normalizar("n chegou!!! tbm ñ veio a nota fiscal")
    assert t.startswith("não chegou ! também")
    assert normalizar(t) == t


def test_titulo_e_mensagem_juntos():
    assert juntar_titulo_mensagem("Ótimo", "Chegou rápido") == "Ótimo. Chegou rápido"
    assert juntar_titulo_mensagem(None, "Chegou rápido") == "Chegou rápido"


def test_informativo_e_contagem_de_palavras():
    assert not informativo("10") and not informativo("...") and not informativo("A")
    assert informativo("ok")
    assert contar_palavras("recomendo . . .") == 1


def test_pii_mascarada_sem_pegar_anos():
    texto, c = mascarar_pii("Me ligue 34 99978 4951 ou (11) 98765-4321. Comprei em 2017 2018")
    assert texto == "Me ligue [TELEFONE] ou [TELEFONE]. Comprei em 2017 2018"
    assert c == {"[TELEFONE]": 2}
    assert mascarar_pii("cpf 123.456.789-09")[0] == "cpf [CPF]"
