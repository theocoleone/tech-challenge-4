import numpy as np
import pandas as pd

import src.index.embeddings as emb
from src.index.bm25 import IndiceBM25, tokenizar
from src.index.vector_store import IndiceVetorial
from src.retrieval.filters import mascara


def test_tokenizar_mantem_negacao_e_tira_acento_e_stop_words():
    assert tokenizar("Não chegou o produto, é uma péssima loja") == ["nao", "chegou", "produto", "pessima", "loja"]


def test_bm25_diferencia_negacao_e_respeita_mascara():
    ids = ["a", "b", "c"]
    idx = IndiceBM25(ids, ["produto não chegou", "produto chegou rápido", "entrega atrasou muito"])
    assert idx.buscar("não chegou", k=1)[0][0] == "a"
    so_c = np.array([False, False, True])
    assert [r for r, _ in idx.buscar("chegou", k=3, mascara=so_c)] == []


def test_filtros_viram_mascara():
    f = {"review_score_max": 2, "categoria_pt": "telefonia", "valor_min": 500}
    docs = pd.DataFrame({"review_score": [1, 5, 2], "categoria_pt": ["telefonia"] * 3, "valor_total": [600.0, 700.0, 100.0]})
    assert mascara(docs, f).tolist() == [True, False, False]
    assert mascara(docs, None).tolist() == [True, True, True]


def test_indice_vetorial_busca_exata_com_e_sem_recorte(tmp_path):
    idx = IndiceVetorial(["a", "b", "c"], np.array([[1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32))
    assert [r for r, _ in idx.buscar(np.array([1, 0.1]), k=2)] == ["a", "b"]
    assert [r for r, _ in idx.buscar(np.array([1, 0.1]), k=2, mascara=np.array([False, True, True]))] == ["b", "c"]
    assert idx.buscar(np.array([1, 0]), k=2, mascara=np.array([False, False, False])) == []
    idx.salvar(tmp_path, "t")
    novo = IndiceVetorial.carregar(tmp_path, "t")
    assert novo.ids == idx.ids and np.allclose(novo.vetores_de(["c"]), [[0, 1]])


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


def test_mmr_evita_quase_duplicatas():
    from src.retrieval.diversity import mmr
    vetores = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.5, 0.5]], dtype=np.float32)  # 0 e 1 quase iguais
    scores = [0.9, 0.89, 0.88, 0.5]
    assert mmr(list("abcd"), scores, vetores, n=2) == [0, 2]  # pula a quase cópia
    assert mmr(list("abcd"), scores, vetores, n=2, lambda_=1.0) == [0, 1]  # só relevância
