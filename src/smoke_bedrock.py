"""Teste de fumaça: confirma que o LLM e o embedding do Bedrock respondem.

Uso: python -m src.smoke_bedrock
"""
from src.config import check_aws_account, get_embeddings, get_llm, get_settings


def main() -> None:
    s = get_settings()
    print(f"profile={s.aws_profile} região={s.aws_region} conta={check_aws_account()}")

    resposta = get_llm().invoke("Responda apenas OK").content
    print(f"LLM {s.llm_model_id}: {resposta!r}")

    vetor = get_embeddings().embed_query("produto chegou atrasado")
    print(f"Embedding {s.emb_model_id}: {len(vetor)} dimensões")


if __name__ == "__main__":
    main()
