"""Serialização estrita dos artefatos de avaliação.

O `json.dumps` do Python aceita NaN por padrão, embora NaN não faça parte do
padrão JSON. Métricas indisponíveis viram `null`, para que qualquer leitor JSON
consiga abrir os resultados.
"""
import json
import math
from pathlib import Path


def _normalizar(valor):
    if isinstance(valor, dict):
        return {chave: _normalizar(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_normalizar(item) for item in valor]
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def salvar_json(arquivo: Path, conteudo: dict) -> None:
    arquivo.write_text(
        json.dumps(_normalizar(conteudo), ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8",
    )
