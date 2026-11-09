import numpy as np
import pandas as pd

import src.index.embeddings as emb
from src.index.bm25 import IndiceBM25, tokenizar
from src.index.build_index import metadados_chroma
from src.retrieval.filters import mascara, para_chroma


def test_tokenizar_mantem_negacao_e_tira_acento_e_stop_words():
    assert tokenizar("Não chegou o produto, é uma péssima loja") == ["nao", "chegou", "produto", "pessima", "loja"]


def test_bm25_diferencia_negacao_e_respeita_mascara():
    ids = ["a", "b", "c"]
    idx = IndiceBM25(ids, ["produto não chegou", "produto chegou rápido", "entrega atrasou muito"])
    assert idx.buscar("não chegou", k=1)[0][0] == "a"
    so_c = np.array([False, False, True])
    assert [r for r, _ in idx.buscar("chegou", k=3, mascara=so_c)] == []


def test_filtros_viram_where_do_chroma_e_mascara():
    f = {"review_score_max": 2, "categoria_pt": "telefonia", "valor_min": 500}
    assert para_chroma(f) == {"$and": [
        {"review_score": {"$lte": 2}}, {"categoria_pt": {"$eq": "telefonia"}}, {"valor_total": {"$gte": 500}}]}
    assert para_chroma({"entregue_com_atraso": True}) == {"entregue_com_atraso": {"$eq": True}}
    assert para_chroma(None) is None
    docs = pd.DataFrame({"review_score": [1, 5, 2], "categoria_pt": ["telefonia"] * 3, "valor_total": [600.0, 700.0, 100.0]})
    assert mascara(docs, f).tolist() == [True, False, False]


def test_metadados_chroma_omite_vazios_e_converte_tipos():
    linha = {k: None for k in ["review_score", "data_avaliacao", "data_envio_pesquisa", "ano_mes_compra", "order_status",
                               "valor_total", "multi_pedido", "entregue", "dias_atraso", "entregue_com_atraso",
                               "categoria_pt", "categoria_en", "multi_categoria", "n_itens", "uf_cliente",
                               "n_palavras", "order_ids"]}
    linha.update(review_score=np.int64(1), entregue=np.bool_(False), dias_atraso=pd.NA, valor_total=np.float64(10.5),
                 categoria_pt="telefonia")
    m = metadados_chroma(linha)
    assert m == {"review_score": 1, "entregue": False, "valor_total": 10.5, "categoria_pt": "telefonia"}
    assert type(m["review_score"]) is int and type(m["entregue"]) is bool


class _Falso:
    chamadas = 0

    def embed_documents(self, textos):
        _Falso.chamadas += len(textos)
        return [[float(len(t)), 1.0] for t in textos]


def test_cache_so_calcula_textos_novos(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "get_embedder", lambda nome: _Falso())
    _Falso.chamadas = 0
    v1 = emb.embeddings_com_cache("fake", ["a", "b"], ["oi", "olá"], tmp_path)
    assert _Falso.chamadas == 2 and v1.shape == (2, 2)
    v2 = emb.embeddings_com_cache("fake", ["a", "b", "c"], ["oi", "olá", "tchau"], tmp_path)
    assert _Falso.chamadas == 3  # só "c" foi calculado
    assert np.allclose(v2[:2], v1)
    emb.embeddings_com_cache("fake", ["a"], ["oi editado"], tmp_path)
    assert _Falso.chamadas == 4  # texto mudou: recalcula
