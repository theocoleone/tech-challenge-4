from types import SimpleNamespace

from src.generation import chain as modulo_chain
from src.generation.chain import (
    _agregar,
    _avaliacoes,
    _finalizar_texto_agregado,
    _insights_agregados_deterministicos,
    _texto_agregado_deterministico,
)
from src.generation.guard import conferir_numeros
from src.generation.schemas import Insight, RespostaVoC


class _Contagem:
    total_com_tema = 10841
    total_no_recorte = 42121
    polaridade = "nota_baixa"
    assunto_nao_modelado = None
    macros = [{"macro": "entrega", "n": 7159, "pct": 66.0}]
    temas = [
        {"tema": "atraso na entrega", "macro": "entrega", "n": 3090, "pct": 28.5},
        {"tema": "produto não entregue", "macro": "entrega", "n": 2083, "pct": 19.2},
    ]


def test_texto_agregado_recebe_base_e_notacao_brasileira():
    texto = _finalizar_texto_agregado(
        "Atraso aparece em 3090 avaliações (28.5%).",
        _Contagem(),
    )
    assert "28,5%" in texto
    assert "10.841 avaliações classificadas (notas 1 ou 2)" in texto
    assert "42.121 avaliações com comentário" in texto


def test_base_nao_e_duplicada_quando_os_dois_totais_ja_aparecem():
    original = (
        "Foram 10841 avaliações classificadas entre 42121 avaliações no recorte."
    )
    assert _finalizar_texto_agregado(original, _Contagem()) == original


def test_concordancia_da_quantidade_de_avaliacoes():
    assert _avaliacoes(1) == "1 avaliação"
    assert _avaliacoes(3) == "3 avaliações"


def test_texto_agregado_deterministico_so_usa_numeros_da_contagem():
    texto = _texto_agregado_deterministico(_Contagem())

    assert "atraso na entrega: 3.090 avaliações (28,5%)" in texto
    assert "produto não entregue: 2.083 avaliações (19,2%)" in texto
    assert conferir_numeros(texto, _Contagem()) == []


def test_insights_deterministicos_usam_apenas_ids_representativos():
    insights = _insights_agregados_deterministicos(
        _Contagem(),
        {
            "atraso na entrega": ["a" * 32, "a" * 32, "b" * 32],
            "produto não entregue": [],
        },
    )

    assert insights == [{
        "afirmacao": 'As avaliações citadas exemplificam o tema "atraso na entrega".',
        "review_ids": ["a" * 32, "b" * 32],
    }]


def test_agregado_usa_fallback_deterministico_apos_duas_respostas_invalidas(monkeypatch):
    review_id = "a" * 32

    class AgregadorFalso:
        def contar(self, *_args, **_kwargs):
            return _Contagem()

        def representativas(self, *_args, **_kwargs):
            return [review_id]

        def contexto(self, *_args, **_kwargs):
            return "<contagem>teste</contagem>"

    class GeradorFalso:
        chamadas = 0

        def invoke(self, _entrada):
            self.chamadas += 1
            return RespostaVoC(
                tem_evidencia=True,
                resposta="Atraso na entrega apareceu em 999 avaliações.",
                insights=[
                    Insight(
                        afirmacao="As avaliações relatam atraso na entrega.",
                        review_ids=[review_id],
                    )
                ],
            )

    gerador = GeradorFalso()
    monkeypatch.setattr(modulo_chain, "_agregador", lambda: AgregadorFalso())
    monkeypatch.setattr(modulo_chain, "_buscador", lambda: SimpleNamespace(por_id=None))
    monkeypatch.setattr(modulo_chain, "_gerador_agregada", lambda: gerador)
    monkeypatch.setattr(
        modulo_chain,
        "_saida",
        lambda estado, status, resposta, insights: {
            "status": status,
            "resposta": resposta,
            "insights": insights,
            "validacao": estado["validacao"],
        },
    )

    resultado = _agregar({
        "pergunta": "Quais são os principais problemas?",
        "filtros": {},
        "temas_alvo": [],
        "tempos": {},
    })

    assert gerador.chamadas == 2
    assert resultado["status"] == "ok"
    assert "999" not in resultado["resposta"]
    assert resultado["validacao"]["fallback_deterministico"] is True
    assert resultado["validacao"]["numeros_rejeitados_pelo_guard"] == ["999"]
    assert resultado["validacao"]["numeros_fora_da_contagem"] == []
    assert resultado["insights"][0]["review_ids"] == [review_id]
