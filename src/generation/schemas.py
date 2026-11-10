"""Estruturas de dados que o LLM precisa preencher (saída estruturada)."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FiltroReviews(BaseModel):
    """Recorte dos metadados pedido na pergunta. Campos vazios = sem filtro."""

    review_score_min: int | None = Field(None, ge=1, le=5, description="nota mínima (ex.: elogios -> 4)")
    review_score_max: int | None = Field(None, ge=1, le=5, description="nota máxima (ex.: insatisfeitos -> 2)")
    categoria_pt: str | None = Field(None, description="categoria exatamente como na lista de categorias")
    ano_mes_compra_min: int | None = Field(None, description="mês inicial da compra, aaaamm")
    ano_mes_compra_max: int | None = Field(None, description="mês final da compra, aaaamm")
    entregue_com_atraso: bool | None = Field(None, description="True só se a pergunta pedir entregas atrasadas")
    order_status: Literal["delivered", "canceled", "shipped", "unavailable", "invoiced", "processing"] | None = None
    uf_cliente: str | None = Field(None, description="sigla do estado, ex.: SP")
    valor_min: float | None = Field(None, description="valor mínimo do pedido em reais")
    valor_max: float | None = Field(None, description="valor máximo do pedido em reais")

    @field_validator("uf_cliente")
    @classmethod
    def _uf_valida(cls, v):
        # o modelo às vezes devolve "RR,SP"; um valor inválido zeraria a busca
        if v is None:
            return None
        v = v.strip().upper()
        return v if len(v) == 2 and v.isalpha() else None

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
