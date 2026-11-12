from src.generation.schemas import AnaliseDaPergunta, FiltroReviews
from src.retrieval.query_transform import corrigir_categoria, tem_assunto_proprio


def _a(consulta, **filtros):
    return AnaliseDaPergunta(escopo="dominio", tipo="especifica", consulta_reescrita=consulta, filtros=FiltroReviews(**filtros))


CATS = ("cama_mesa_banho", "eletronicos", "informatica_acessorios")


def test_categoria_nao_citada_na_pergunta_e_descartada():
    a = corrigir_categoria("O que os clientes acham dos drones?", _a("drones", categoria_pt="eletronicos"), CATS)
    assert a.filtros.categoria_pt is None
    a = corrigir_categoria("Problemas em cama_mesa_banho?", _a("problema", categoria_pt="cama_mesa_banho"), CATS)
    assert a.filtros.categoria_pt == "cama_mesa_banho"
    a = corrigir_categoria("Problemas em cama, mesa e banho?", _a("problema", categoria_pt="cama_mesa_banho"), CATS)
    assert a.filtros.categoria_pt == "cama_mesa_banho"


def test_categoria_fora_da_lista_vira_a_mais_parecida_ou_some():
    a = corrigir_categoria("Problemas em cama mesa e banho?", _a("problema", categoria_pt="cama_mesa_e_banho"), CATS)
    assert a.filtros.categoria_pt == "cama_mesa_banho"
    a = corrigir_categoria("Problemas com brinquedos?", _a("problema", categoria_pt="brinquedos_infantis"), CATS)
    assert a.filtros.categoria_pt is None


def test_pergunta_sem_assunto_proprio():
    assert not tem_assunto_proprio(_a("problema reclamação cama mesa banho", categoria_pt="cama_mesa_banho"))
    assert not tem_assunto_proprio(_a("problema reclamação", valor_min=500))
    assert not tem_assunto_proprio(_a("elogios informatica_acessorios", categoria_pt="informatica_acessorios"))
    assert tem_assunto_proprio(_a("troca devolver problema"))
    assert tem_assunto_proprio(_a("embalagem"))
