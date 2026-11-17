from src.generation.guard import limpar_ids, validar

ID_A = "a" * 32
ID_B = "b" * 32


def test_citacao_fora_do_contexto_e_removida():
    insights = [{"afirmacao": "x", "review_ids": [ID_A, "inventado"]}, {"afirmacao": "y", "review_ids": ["outro"]}]
    v = validar("ok", insights, {ID_A, ID_B})
    assert v.insights == [{"afirmacao": "x", "review_ids": [ID_A]}]
    assert v.citacoes_invalidas == 2


def test_id_escrito_na_afirmacao_e_removido():
    v = validar(
        "ok",
        [{"afirmacao": f"Atraso relatado no ID: {ID_A}.", "review_ids": [ID_A]}],
        {ID_A},
    )
    assert v.insights == [{"afirmacao": "Atraso relatado no.", "review_ids": [ID_A]}]


def test_detecta_linguagem_de_frequencia():
    assert validar("A maioria dos clientes reclama", [], set()).linguagem_de_frequencia == ["A maioria"]
    assert validar("Cerca de 40% reclamam", [], set()).linguagem_de_frequencia == ["40%"]
    assert validar("Alguns clientes relatam atraso", [], set()).linguagem_de_frequencia == []


def test_remove_ids_do_texto():
    texto = f"Clientes relatam atraso (ID: {ID_A}). Outros, como {ID_B}, falam de demora."
    assert limpar_ids(texto) == "Clientes relatam atraso. Outros, como falam de demora."


class _Contagem:
    total_no_recorte, total_com_tema = 42121, 10841
    polaridade = "nota_baixa"
    macros = [{"macro": "entrega", "n": 7159, "pct": 66.0}]
    temas = [{"tema": "atraso na entrega", "macro": "entrega", "n": 3090, "pct": 28.5}]


class _ContagemDoisTemas(_Contagem):
    temas = [
        {"tema": "atraso na entrega", "macro": "entrega", "n": 3090, "pct": 28.5},
        {"tema": "produto não entregue", "macro": "entrega", "n": 2083, "pct": 19.2},
    ]


def test_numeros_fora_da_contagem_sao_apontados():
    from src.generation.guard import conferir_numeros

    ok = "De 10841 avaliações com nota 1 ou 2, atraso na entrega aparece em 3090 (28,5%)."
    assert conferir_numeros(ok, _Contagem()) == []
    assert conferir_numeros("Foram 10.841 avaliações; atraso apareceu em 3.090 (28,5%).", _Contagem()) == []
    inventado = "Cerca de 45% dos clientes reclamam de atraso, em 5000 avaliações."
    assert set(conferir_numeros(inventado, _Contagem())) == {"45%", "5000"}
    assert conferir_numeros("Cerca de 5% dos clientes reclamam.", _Contagem()) == ["5%"]


def test_numeros_precisam_pertencer_ao_tema_citado():
    from src.generation.guard import conferir_numeros

    trocou_contagem = "Atraso na entrega apareceu em 7159 avaliações (28,5%)."
    trocou_percentual = "Atraso na entrega apareceu em 3090 avaliações (66%)."
    assert conferir_numeros(trocou_contagem, _Contagem()) == ["7159"]
    assert conferir_numeros(trocou_percentual, _Contagem()) == ["66%"]


def test_percentual_fica_com_rotulo_anterior_mesmo_perto_do_proximo():
    from src.generation.guard import conferir_numeros

    correto = (
        "Atraso na entrega: 3.090 avaliações (28,5%); "
        "produto não entregue: 2.083 avaliações (19,2%)."
    )
    trocado = (
        "Atraso na entrega: 2.083 avaliações (19,2%); "
        "produto não entregue: 3.090 avaliações (28,5%)."
    )
    assert conferir_numeros(correto, _ContagemDoisTemas()) == []
    assert set(conferir_numeros(trocado, _ContagemDoisTemas())) == {
        "2.083", "19,2%", "3.090", "28,5%",
    }


def test_prefixo_numerico_de_review_id_nao_e_contagem():
    from src.generation.guard import conferir_numeros

    review_id = "2995bd97f892c55e78dd69147e45e6fc"
    texto = f"Atraso na entrega: 3.090 avaliações (28,5%), conforme {review_id}."
    assert conferir_numeros(texto, _Contagem()) == []


def test_valor_de_filtro_exige_contexto_monetario():
    from src.generation.guard import conferir_numeros

    filtros = {"valor_min": 500}
    assert conferir_numeros("Foram 500 avaliações.", _Contagem(), filtros) == ["500"]
    assert conferir_numeros("Em pedidos acima de R$ 500, foram 10841 avaliações.", _Contagem(), filtros) == []
