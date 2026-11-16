"""Confere se cada afirmação da resposta é sustentada pelas avaliações citadas.

O juiz é um modelo diferente do que escreve as respostas (Nova Lite julga o que o
Nova Pro escreveu): usar o mesmo modelo nos dois papéis tende a aprovar o próprio
texto. O resultado automático deve ser complementado por auditoria humana antes
de ser tratado como estimativa independente de fidelidade.
"""
from functools import lru_cache

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import ROOT, get_llm, get_settings


class Veredito(BaseModel):
    suportada: bool = Field(description="a afirmação é sustentada pelas avaliações citadas?")
    motivo: str = Field(description="uma frase curta")


@lru_cache
def _juiz():
    prompt = ChatPromptTemplate.from_messages([
        ("system", (ROOT / "src" / "generation" / "prompts" / "juiz_v1.md").read_text()),
        ("human", "Pergunta do gestor: {pergunta}\n\nAfirmação: {afirmacao}\n\nAvaliações citadas:\n{avaliacoes}"),
    ])
    return prompt | get_llm(model_id=get_settings().judge_model_id, estruturado=Veredito)


def julgar(pergunta: str, afirmacao: str, textos: list[str]) -> Veredito:
    return _juiz().invoke({"pergunta": pergunta, "afirmacao": afirmacao,
                           "avaliacoes": "\n".join(f"- {t}" for t in textos)})


def fidelidade(respostas: list[dict], por_id: pd.DataFrame) -> tuple[float, list[dict]]:
    """Fração das afirmações sustentadas, entre as respostas que citaram evidências."""
    vereditos = []
    for r in respostas:
        for ins in r.get("insights", []):
            textos = [por_id.texto_original[rid] for rid in ins["review_ids"] if rid in por_id.index]
            if not textos:
                continue
            v = julgar(r["pergunta"], ins["afirmacao"], textos)
            vereditos.append({"pergunta_id": r.get("id"), "afirmacao": ins["afirmacao"],
                              "review_ids": ins["review_ids"], "suportada": v.suportada, "motivo": v.motivo})
    apoiadas = sum(v["suportada"] for v in vereditos)
    return (apoiadas / len(vereditos) if vereditos else float("nan")), vereditos
