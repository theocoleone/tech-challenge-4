import pandas as pd

from src.analytics.aggregate import Agregador


def _agregador(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.rerank.reordenar",
        lambda pergunta, pares: [(rid, 1.0) for rid, _ in pares],
    )
    docs = pd.DataFrame({
        "review_id": list("abcdef"), "review_score": [1, 1, 2, 5, 5, 4],
        "categoria_pt": ["telefonia"] * 3 + ["bebes"] * 3,
        "texto_original": ["entrega atrasou muito", "chegou com atraso", "veio quebrado",
                           "chegou rápido", "entrega veloz", "produto excelente"],
    })
    temas = pd.DataFrame({
        "review_id": list("abcdef"),
        "grupo": ["nota_baixa"] * 3 + ["nota_alta"] * 3,
        "tema_id": ["nota_baixa_00", "nota_baixa_01", "nota_baixa_02", "nota_alta_00", "nota_alta_00", "nota_alta_01"],
        "nome_tema": ["atraso na entrega", "atraso na entrega", "produto quebrado",
                      "entrega rápida", "entrega rápida", "produto de qualidade"],
        "macro": ["entrega", "entrega", "produto", "entrega", "entrega", "produto"],
        "dist_centro": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
    })
    ag = Agregador.__new__(Agregador)
    ag.docs, ag.temas = docs, temas
    ag.textos = docs.set_index("review_id").texto_original
    return ag


def test_polaridade_vem_do_filtro_ou_das_palavras(monkeypatch):
    ag = _agregador(monkeypatch)
    assert ag.polaridade("o que mais elogiam?", {}) == "nota_alta"
    assert ag.polaridade("quais os principais problemas?", {}) == "nota_baixa"
    assert ag.polaridade("o que dizem?", {"review_score_min": 4}) == "nota_alta"
    assert ag.polaridade("o que elogiam?", {"review_score_max": 2}) == "nota_baixa"
    assert ag.polaridade("quais temas aparecem?", {}) == "todas"
    assert ag.polaridade(
        "quais temas aparecem nos pedidos entregues com atraso?",
        {"entregue_com_atraso": True},
    ) == "todas"


def test_contagem_usa_a_base_inteira_do_recorte(monkeypatch):
    ag = _agregador(monkeypatch)
    c = ag.contar("principais problemas", {})
    assert c.polaridade == "nota_baixa" and c.total_com_tema == 3
    assert c.temas[0] == {"tema": "atraso na entrega", "macro": "entrega", "n": 2, "pct": 66.7}
    filtrada = ag.contar("principais problemas", {"categoria_pt": "telefonia"})
    assert filtrada.total_no_recorte == 3


def test_assunto_restringe_so_quando_aponta_um_macro(monkeypatch):
    ag = _agregador(monkeypatch)
    só_entrega = ag.contar("reclamações de entrega", {}, ["entrega"])
    assert só_entrega.total_com_tema == 2 and {m["macro"] for m in só_entrega.macros} == {"entrega"}
    geral = ag.contar("principais problemas", {}, ["problemas"])
    assert geral.total_com_tema == 3  # termo genérico: não restringe


def test_assunto_que_o_catalogo_nao_modela_nao_cai_na_contagem_global(monkeypatch):
    ag = _agregador(monkeypatch)
    c = ag.contar("principais reclamações de atendimento", {}, ["atendimento"])
    assert c.total_com_tema == 0
    assert c.assunto_nao_modelado == "atendimento"


def test_tema_existente_mas_ausente_no_recorte_nao_vira_nao_modelado(monkeypatch):
    ag = _agregador(monkeypatch)
    ag.temas.loc[ag.temas.review_id == "f", ["nome_tema", "macro"]] = [
        "Devolução e reembolso", "troca e devolução"
    ]
    c = ag.contar("trocas", {"categoria_pt": "telefonia"}, ["troca"])
    assert c.total_com_tema == 0
    assert c.assunto_nao_modelado is None


def test_taxonomia_controlada_nao_confunde_assuntos_de_produto():
    catalogo = pd.DataFrame({
        "nome_tema": [
            "produto de qualidade", "produto não entregue", "Problemas com nota fiscal",
            "Devolução e reembolso",
        ],
        "macro": ["produto", "entrega", "compra em geral", "troca e devolução"],
    })
    selecionar = Agregador._selecionar_alvo
    assert selecionar(catalogo, ["trocas de produtos"])[0] == {
        "macro": {"troca e devolução"}
    }
    assert selecionar(catalogo, ["produto não chegou"])[0] == {"macro": {"entrega"}}
    assert selecionar(catalogo, ["fatura e nota fiscal"])[0] == {
        "nome_tema": {"Problemas com nota fiscal"}
    }


def test_representativas_sao_reordenadas_pelo_nome_do_tema(monkeypatch):
    ag = _agregador(monkeypatch)
    # "entrega atrasou muito" e "chegou com atraso" são do tema; o re-ranker escolhe
    assert ag.representativas("atraso na entrega", {}, n=1)[0] in {"a", "b"}
    assert ag.representativas("atraso na entrega", {"categoria_pt": "bebes"}) == []


def test_representativas_respeitam_a_polaridade_contada(monkeypatch):
    ag = _agregador(monkeypatch)
    ag.temas.loc[ag.temas.review_id == "d", "nome_tema"] = "atraso na entrega"
    ids = ag.representativas("atraso na entrega", {}, polaridade="nota_baixa")
    assert set(ids) <= {"a", "b"}
