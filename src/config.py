"""Configuração central do TC4 e fábrica de provedores (LLM e embeddings).

Tudo que varia entre ambientes vem de variáveis de ambiente com prefixo `TC4_`
ou do `.env` (ver `.env.example`). O resto do código pede o modelo a `get_llm()` e
`get_embeddings()` e não sabe qual provedor está por trás, então trocar
Nova por Haiku, ou Bedrock pelo modo local, é só mudar a configuração.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Prefixo TC4_: variáveis globais do shell (ex.: AWS_REGION do ambiente de trabalho)
    # não vazam para o projeto.
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="TC4_", extra="ignore")

    aws_profile: str = "fiap-tech-challenge"
    aws_region: str = "us-east-1"
    # Conta pessoal do curso. Qualquer outra (ex.: contas de trabalho) aborta a execução.
    aws_account_id: str = "286958704145"

    llm_provider: Literal["bedrock", "local"] = "bedrock"
    emb_provider: Literal["bedrock", "local"] = "bedrock"

    llm_model_id: str = "us.amazon.nova-lite-v1:0"
    # Reserva usada quando o modelo principal demora ou é limitado (throttling)
    llm_fallback_model_id: str = "us.amazon.nova-pro-v1:0"
    llm_timeout_s: int = 20
    judge_model_id: str = "us.amazon.nova-pro-v1:0"
    emb_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_temperature: float = 0.0

    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    # Pasta dos índices (BM25, vetores e cache de embeddings); pode ficar fora do repo.
    index_dir: Path = ROOT / "index"

    @field_validator("data_raw", "data_processed", "index_dir", mode="before")
    @classmethod
    def _expandir_til(cls, v):
        return Path(v).expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()


class ContaAwsErrada(RuntimeError):
    """A credencial resolvida não pertence à conta esperada do projeto."""


@lru_cache
def check_aws_account() -> str:
    """Confere, uma vez por processo, que a credencial é da conta do curso.

    Fail closed: se a conta for outra, ou se não der para descobrir, nenhuma
    chamada ao Bedrock acontece.
    """
    import boto3

    s = get_settings()
    session = boto3.Session(profile_name=s.aws_profile, region_name=s.aws_region)
    conta = session.client("sts").get_caller_identity()["Account"]
    if conta != s.aws_account_id:
        raise ContaAwsErrada(
            f"profile '{s.aws_profile}' resolveu para a conta {conta}, "
            f"mas o TC4 só roda na conta {s.aws_account_id}"
        )
    return conta


def _chat_bedrock(model_id: str):
    from botocore.config import Config
    from langchain_aws import ChatBedrockConverse

    s = get_settings()
    return ChatBedrockConverse(
        model_id=model_id,
        region_name=s.aws_region,
        credentials_profile_name=s.aws_profile,
        temperature=s.llm_temperature,
        # Sem isto, uma resposta lenta do Bedrock espera 60 s e é repetida várias vezes
        # (vimos chamadas de 3 a 9 minutos). Melhor falhar rápido e ir para a reserva.
        config=Config(read_timeout=s.llm_timeout_s, connect_timeout=5,
                      retries={"mode": "adaptive", "max_attempts": 2}),
    )


def get_llm(model_id: str | None = None, estruturado=None):
    """LLM do projeto. Com `estruturado` (um modelo pydantic), devolve saída estruturada.

    O modelo principal tem uma reserva (`llm_fallback_model_id`) para quando ele
    demora ou é limitado; o trace registra qual modelo respondeu.
    """
    s = get_settings()
    if s.llm_provider == "local":
        raise NotImplementedError("LLM local (plano B2) entra só se o Bedrock continuar bloqueado")
    check_aws_account()
    principal = _chat_bedrock(model_id or s.llm_model_id)
    if model_id:  # modelo escolhido explicitamente (ex.: o juiz): sem reserva
        return principal.with_structured_output(estruturado) if estruturado else principal
    reserva = _chat_bedrock(s.llm_fallback_model_id)
    if estruturado:
        return principal.with_structured_output(estruturado).with_fallbacks([reserva.with_structured_output(estruturado)])
    return principal.with_fallbacks([reserva])


def get_embeddings():
    s = get_settings()
    if s.emb_provider == "local":
        raise NotImplementedError("Embeddings locais (plano B2) entram só se o Bedrock continuar bloqueado")
    check_aws_account()
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(
        model_id=s.emb_model_id,
        region_name=s.aws_region,
        credentials_profile_name=s.aws_profile,
    )
