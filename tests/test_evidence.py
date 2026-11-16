import pandas as pd
import pytest

from src.retrieval.evidence import (
    VerificadorEvidenciaLexical,
    classificar_quantidade,
    normalizar_lexico,
)


def _documentos():
    return pd.DataFrame({
        "review_id": list("abcdefgh"),
        "texto_normalizado": [
            "Esqueci minha senha e não consegui entrar.",
            "Esta é apenas uma resenha do produto.",
            "O drone não funciona.",
            "Comprei dois drones para presente.",
            "A devolução foi solicitada.",
            "As devoluções ainda não foram processadas.",
            "A nota fiscal veio errada.",
            "Recebi a nota, mas não era um documento fiscal.",
        ],
        "review_score": [1, 5, 1, 2, 1, 5, 1, 5],
        "uf_cliente": ["SP", "SP", "RR", "SP", "SP", "SP", "RR", "SP"],
        "valor_total": [100.0, 100.0, 600.0, 50.0, 700.0, 800.0, 20.0, 30.0],
    })


@pytest.mark.parametrize(
    ("quantidade", "esperado"),
    [
        (0, "sem_evidencia"),
        (1, "evidencia_fraca"),
        (3, "evidencia_fraca"),
        (4, "evidencia_suficiente"),
        (100, "evidencia_suficiente"),
    ],
)
def test_classifica_zero_um_a_tres_e_quatro_ou_mais(quantidade, esperado):
    assert classificar_quantidade(quantidade) == esperado


def test_normalizacao_remove_acentos_caixa_e_pontuacao():
    assert normalizar_lexico("  DEVOLUÇÃO — Não feita! ") == "devolucao nao feita"


def test_limites_de_palavra_nao_confundem_senha_com_resenha():
    suporte = VerificadorEvidenciaLexical(_documentos()).avaliar(
        "Quais problemas existem com senha?", temas_alvo=["senha"]
    )
    assert suporte.quantidade == 1
    assert suporte.review_ids == ("a",)
    assert suporte.faixa == "evidencia_fraca"


def test_variacoes_simples_de_plural_preservam_limites():
    verificador = VerificadorEvidenciaLexical(_documentos())
    drones = verificador.avaliar("O que acham de drones?", temas_alvo=["drone"])
    devolucoes = verificador.avaliar("Como são as devoluções?", temas_alvo=["devolução"])
    assert drones.review_ids == ("c", "d")
    assert devolucoes.review_ids == ("e", "f")


def test_itens_de_temas_sao_alternativas_e_palavras_do_item_sao_conjuncao():
    verificador = VerificadorEvidenciaLexical(_documentos())
    alternativas = verificador.avaliar("", temas_alvo=["drone", "devolução"])
    alternativas_no_mesmo_item = verificador.avaliar("", temas_alvo=["drone, devolução"])
    nota_fiscal = verificador.avaliar("", temas_alvo=["nota fiscal"])
    assert alternativas.review_ids == ("c", "d", "e", "f")
    assert alternativas_no_mesmo_item.review_ids == alternativas.review_ids
    assert alternativas.faixa == "evidencia_suficiente"
    assert nota_fiscal.review_ids == ("g", "h")


def test_filtros_restringem_contagem_e_ids_antes_da_classificacao():
    suporte = VerificadorEvidenciaLexical(_documentos()).avaliar(
        "O que relatam sobre drones?",
        temas_alvo=["drone"],
        filtros={"review_score_max": 1, "uf_cliente": "RR", "valor_min": 500},
    )
    assert suporte.total_no_recorte == 1
    assert suporte.review_ids == ("c",)
    assert suporte.faixa == "evidencia_fraca"


def test_fallback_da_pergunta_entende_alternativas_sem_casar_substring():
    suporte = VerificadorEvidenciaLexical(_documentos()).avaliar(
        "Quais problemas os clientes relatam com login ou senha no site?"
    )
    assert suporte.origem_termos == "pergunta"
    assert suporte.grupos_de_termos == (("login",), ("senha",))
    assert suporte.review_ids == ("a",)


def test_termo_especifico_prevalece_sobre_modificador_de_dominio():
    docs = pd.DataFrame({
        "review_id": ["w1", "w2", "j1", "p1"],
        "texto_original": [
            "Enviei mensagem pelo WhatsApp e ninguém respondeu.",
            "O atendimento demorou.",
            "A cobrança de juros foi indevida.",
            "Escolhi o parcelamento em três vezes.",
        ],
    })
    verificador = VerificadorEvidenciaLexical(docs)

    whatsapp = verificador.avaliar(
        "Como avaliam o atendimento pelo WhatsApp?",
        temas_alvo=["atendimento WhatsApp"],
    )
    whatsapp_separado = verificador.avaliar(
        "Como avaliam o atendimento pelo WhatsApp?",
        temas_alvo=["atendimento", "WhatsApp"],
    )
    juros = verificador.avaliar(
        "Há cobrança de juros no parcelamento?",
        temas_alvo=["juros no parcelamento"],
    )
    juros_da_pergunta = verificador.avaliar(
        "Há cobrança de juros no parcelamento?"
    )

    assert whatsapp.grupos_de_termos == (("whatsapp",),)
    assert whatsapp.review_ids == ("w1",)
    assert whatsapp_separado.review_ids == whatsapp.review_ids
    assert juros.grupos_de_termos == (("juros",),)
    assert juros.review_ids == ("j1",)
    assert juros_da_pergunta.review_ids == juros.review_ids


def test_modificador_de_dominio_e_preservado_quando_e_o_unico_assunto():
    docs = pd.DataFrame({
        "review_id": ["a", "b"],
        "texto_original": ["O atendimento demorou.", "O produto chegou."],
    })
    suporte = VerificadorEvidenciaLexical(docs).avaliar(
        "O que reclamam do atendimento?", temas_alvo=["atendimento"]
    )
    assert suporte.grupos_de_termos == (("atendimento",),)
    assert suporte.review_ids == ("a",)


def test_aceita_um_unico_tema_como_string_sem_iterar_caracteres():
    suporte = VerificadorEvidenciaLexical(_documentos()).avaliar(
        "Como são os drones?", temas_alvo="drone"
    )
    assert suporte.grupos_de_termos == (("drone",),)
    assert suporte.review_ids == ("c", "d")


def test_pergunta_generica_nao_e_tratada_como_assunto_ausente():
    suporte = VerificadorEvidenciaLexical(_documentos()).avaliar(
        "Quais são os principais problemas relatados pelos clientes?"
    )
    assert not suporte.aplicavel
    assert suporte.faixa is None
    assert suporte.origem_termos == "nenhuma"


def test_carrega_documents_parquet(tmp_path):
    caminho = tmp_path / "documents.parquet"
    _documentos().to_parquet(caminho, index=False)
    suporte = VerificadorEvidenciaLexical.de_parquet(caminho).avaliar(
        "Existe cashback?", temas_alvo=["cashback"], filtros={"uf_cliente": "SP"}
    )
    assert suporte.aplicavel
    assert suporte.quantidade == 0
    assert suporte.faixa == "sem_evidencia"
    assert suporte.total_no_recorte == 6


def test_recusa_quantidade_negativa_e_documentos_sem_colunas_obrigatorias():
    with pytest.raises(ValueError):
        classificar_quantidade(-1)
    with pytest.raises(ValueError):
        VerificadorEvidenciaLexical(pd.DataFrame({"texto_original": ["x"]}))
