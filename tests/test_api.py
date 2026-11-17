from fastapi.testclient import TestClient


def _cliente():
    from src.api.main import app

    return TestClient(app)


def test_health_descreve_o_servico():
    r = _cliente().get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["documentos"] > 40000 and d["temas"] > 5
    assert d["modelos"]["embedding"].startswith("amazon.titan-embed")


def test_review_inexistente_devolve_404():
    assert _cliente().get("/reviews/nao-existe").status_code == 404


def test_review_existente_traz_metadados():
    from src.api.main import _documentos

    por_id = _documentos().set_index("review_id")
    rid = por_id.index[por_id.dias_atraso.isna()][0]  # pedido sem entrega: campo nulo
    d = _cliente().get(f"/reviews/{rid}").json()
    assert d["review_id"] == rid and d["dias_atraso"] is None
    assert isinstance(d["texto_original"], str) and isinstance(d["review_score"], int)


def test_themes_conta_sobre_a_base_e_aceita_filtro():
    c = _cliente()
    geral = c.get("/themes?polaridade=nota_baixa&top=3").json()
    assert geral["total_com_tema"] > 1000 and len(geral["temas"]) == 3
    assert sum(t["n"] for t in geral["macros"]) == geral["total_com_tema"]
    filtrado = c.get("/themes?polaridade=nota_baixa&categoria_pt=telefonia").json()
    assert filtrado["total_no_recorte"] < geral["total_no_recorte"]


def test_categorias_sao_listadas():
    cats = _cliente().get("/categorias").json()
    assert "cama_mesa_banho" in cats and cats == sorted(cats)


def test_filtro_desconhecido_e_intervalo_invertido_sao_recusados():
    c = _cliente()
    desconhecido = c.post("/ask", json={"pergunta": "O que relatam?", "filtros": {"foo": 1}})
    invertido = c.post(
        "/ask",
        json={"pergunta": "O que relatam?", "filtros": {"review_score_min": 5, "review_score_max": 1}},
    )
    assert desconhecido.status_code == 422
    assert invertido.status_code == 422


def test_filtros_manuais_invalidos_sao_recusados():
    c = _cliente()
    uf = c.post("/ask", json={"pergunta": "O que relatam?", "filtros": {"uf_cliente": "S1"}})
    categoria = c.post(
        "/ask", json={"pergunta": "O que relatam?", "filtros": {"categoria_pt": "categoria_inexistente"}}
    )
    assert uf.status_code == 422
    assert categoria.status_code == 422


def test_search_recusa_categoria_desconhecida():
    resposta = _cliente().post(
        "/search",
        json={"pergunta": "O que relatam?", "filtros": {"categoria_pt": "categoria_inexistente"}},
    )
    assert resposta.status_code == 422


def test_themes_reutiliza_a_validacao_dos_filtros():
    c = _cliente()
    urls = [
        "/themes?ano_mes_compra_min=201813",
        "/themes?review_score_min=5&review_score_max=1",
        "/themes?categoria_pt=categoria_inexistente",
        "/themes?polaridade=nota_baixa&review_score_min=5",
    ]
    assert [c.get(url).status_code for url in urls] == [422, 422, 422, 422]


def test_acesso_de_fora_da_maquina_e_recusado():
    c = TestClient(_cliente().app, client=("203.0.113.7", 50000))
    assert c.get("/health").status_code == 403
    assert "local" in c.get("/health").json()["detail"]
