from src.generation.schemas import AnaliseDaPergunta, FiltroReviews
from src.generation.chain import _mesclar_filtros
from src.retrieval.query_transform import (
    corrigir_categoria,
    pede_comparacao,
    tem_assunto_proprio,
    temas_explicitos,
)


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


def test_filtro_manual_remove_a_ponta_extraida_que_ficou_contraditoria():
    assert _mesclar_filtros({"review_score_min": 4}, {"review_score_max": 2}) == {
        "review_score_max": 2
    }
    assert _mesclar_filtros({"valor_max": 100}, {"valor_min": 500}) == {
        "valor_min": 500
    }


def test_comparacao_entre_recortes_e_detectada_sem_confundir_ranking():
    assert pede_comparacao("Roraima reclama mais de atraso do que São Paulo?")
    assert pede_comparacao("Compare as reclamações de RR e SP")
    assert not pede_comparacao("Quais são os problemas mais frequentes?")
    assert not pede_comparacao("O que os clientes mais elogiam?")


def test_tema_omitido_e_recuperado_da_pergunta_original():
    a = _a("padrões qualidade produto")
    assert temas_explicitos(
        "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
        a,
    ) == ["qualidade dos produtos"]


def test_tema_que_ja_virou_filtro_nao_restringe_a_contagem():
    a = _a("entrega atrasada demorou para chegar", entregue_com_atraso=True)
    a.temas_alvo = ["entrega atrasada"]
    assert temas_explicitos(
        "Quais temas aparecem nas avaliações de pedidos entregues com atraso?",
        a,
    ) == []


def test_tema_generico_ou_status_ja_filtrado_e_descartado():
    generico = _a("problema", review_score_max=1)
    generico.temas_alvo = ["problemas nos pedidos"]
    assert temas_explicitos("Quais problemas aparecem?", generico) == []

    cancelados = _a("problema", order_status="canceled")
    cancelados.temas_alvo = ["problemas em pedidos cancelados"]
    assert temas_explicitos(
        "Quais problemas aparecem em pedidos cancelados?",
        cancelados,
    ) == []
