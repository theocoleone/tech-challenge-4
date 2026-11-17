import pandas as pd

from src.eval.check_golden import conferir


def test_termos_de_verificacao_respeitam_limites_de_palavra():
    docs = pd.DataFrame(
        {
            "texto_normalizado": [
                "li várias resenhas antes da compra",
                "esqueci minha senha no site",
                "não consigo fazer login",
            ]
        }
    )
    pergunta = {
        "id": "I02",
        "status_esperado": "evidencia_fraca",
        "termos_verificacao": [r"\b(?:login|senha)\b"],
    }

    assert conferir([pergunta], docs) == []
    assert pergunta["_ocorrencias"] == 2
