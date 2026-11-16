import json

import numpy as np

from src.eval.json_utils import salvar_json


def test_salvar_json_converte_nao_finitos_e_tipos_numpy(tmp_path):
    arquivo = tmp_path / "resultado.json"

    salvar_json(
        arquivo,
        {
            "nan": float("nan"),
            "infinito": np.float64("inf"),
            "inteiro": np.int64(7),
            "tupla": (1, 2),
        },
    )

    resultado = json.loads(arquivo.read_text(encoding="utf-8"))
    assert resultado == {
        "nan": None,
        "infinito": None,
        "inteiro": 7,
        "tupla": [1, 2],
    }
