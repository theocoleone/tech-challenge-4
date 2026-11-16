"""Estruturas de dados que o LLM precisa preencher (saída estruturada)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UFS_BRASIL = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}


class FiltroReviews(BaseModel):
    """Recorte dos metadados pedido na pergunta. Campos vazios = sem filtro."""

    model_config = ConfigDict(extra="forbid")

    review_score_min: int | None = Field(None, ge=1, le=5, description="nota mínima (ex.: elogios -> 4)")
    review_score_max: int | None = Field(None, ge=1, le=5, description="nota máxima (ex.: insatisfeitos -> 2)")
    categoria_pt: str | None = Field(None, description="categoria exatamente como na lista de categorias")
    ano_mes_compra_min: int | None = Field(None, description="mês inicial da compra, aaaamm")
    ano_mes_compra_max: int | None = Field(None, description="mês final da compra, aaaamm")
    entregue_com_atraso: bool | None = Field(None, description="True só se a pergunta pedir entregas atrasadas")
    order_status: Literal[
        "delivered", "canceled", "shipped", "unavailable", "invoiced", "processing", "created"
    ] | None = Field(None, description="status do pedido em inglês conforme a base Olist")
    uf_cliente: str | None = Field(None, description="sigla do estado, ex.: SP")
    valor_min: float | None = Field(None, ge=0, description="valor mínimo do pedido em reais")
    valor_max: float | None = Field(None, ge=0, description="valor máximo do pedido em reais")

    @field_validator("uf_cliente")
    @classmethod
    def _uf_valida(cls, v):
        if v is None:
            return None
        v = v.strip().upper()
        if v not in UFS_BRASIL:
            raise ValueError("UF deve ser uma sigla brasileira válida")
        return v

    @field_validator("ano_mes_compra_min", "ano_mes_compra_max")
    @classmethod
    def _ano_mes_valido(cls, v):
        if v is None:
            return None
        ano, mes = divmod(v, 100)
        if not 1900 <= ano <= 2100 or not 1 <= mes <= 12:
            raise ValueError("mês deve estar no formato aaaamm")
        return v

    @model_validator(mode="after")
    def _intervalos_validos(self):
        for minimo, maximo, nome in (
            (self.review_score_min, self.review_score_max, "nota"),
            (self.ano_mes_compra_min, self.ano_mes_compra_max, "período"),
            (self.valor_min, self.valor_max, "valor"),
        ):
            if minimo is not None and maximo is not None and minimo > maximo:
                raise ValueError(f"limite mínimo de {nome} maior que o máximo")
        return self

    def como_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class AnaliseDaPergunta(BaseModel):
    escopo: Literal["dominio", "fora_escopo"] = Field(
        description="dominio: pergunta sobre a experiência de compra dos clientes (produtos, entrega, loja, "
        "atendimento, pagamento, avaliações). fora_escopo: qualquer outro assunto.")
    tipo: Literal["especifica", "agregada"] = Field(
        description="agregada: pede o que é mais frequente, principais temas ou padrões gerais. "
        "especifica: pede relatos ou evidências sobre um assunto.")
    consulta_reescrita: str = Field(
        description="a pergunta reescrita como uma busca curta, com os termos que os clientes usariam nas avaliações")
    filtros: FiltroReviews = Field(default_factory=FiltroReviews)
    temas_alvo: list[str] = Field(default_factory=list, description="assuntos citados na pergunta, ex.: ['entrega']")



class Insight(BaseModel):
    afirmacao: str = Field(description="uma conclusão curta sustentada pelas avaliações")
    review_ids: list[str] = Field(description="IDs das avaliações que sustentam a afirmação (1 a 3)")


class RespostaVoC(BaseModel):
    tem_evidencia: bool = Field(description="False se as avaliações fornecidas não respondem à pergunta")
    resposta: str = Field(description="resposta em português, em 2 a 5 frases, só com o que está nas avaliações")
    insights: list[Insight] = Field(default_factory=list)
