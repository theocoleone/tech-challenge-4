from src.generation.guard import limpar_ids, validar

ID_A = "a" * 32
ID_B = "b" * 32


def test_citacao_fora_do_contexto_e_removida():
    insights = [{"afirmacao": "x", "review_ids": [ID_A, "inventado"]}, {"afirmacao": "y", "review_ids": ["outro"]}]
    v = validar("ok", insights, {ID_A, ID_B})
    assert v.insights == [{"afirmacao": "x", "review_ids": [ID_A]}]
    assert v.citacoes_invalidas == 2


def test_detecta_linguagem_de_frequencia():
    assert validar("A maioria dos clientes reclama", [], set()).linguagem_de_frequencia == ["A maioria"]
    assert validar("Cerca de 40% reclamam", [], set()).linguagem_de_frequencia == ["40%"]
    assert validar("Alguns clientes relatam atraso", [], set()).linguagem_de_frequencia == []


def test_remove_ids_do_texto():
    texto = f"Clientes relatam atraso (ID: {ID_A}). Outros, como {ID_B}, falam de demora."
    assert limpar_ids(texto) == "Clientes relatam atraso. Outros, como falam de demora."
