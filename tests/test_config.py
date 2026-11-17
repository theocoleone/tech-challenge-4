from src.config import Settings


def test_defaults_usam_bedrock_e_conta_do_curso(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")  # global do shell não pode vazar
    s = Settings(_env_file=None)
    assert s.llm_provider == "bedrock"
    assert s.emb_provider == "bedrock"
    assert s.aws_profile == "fiap-tech-challenge"
    assert s.aws_region == "us-east-1"
    assert s.aws_account_id == ""


def test_variavel_de_ambiente_troca_o_modelo(monkeypatch):
    monkeypatch.setenv("TC4_LLM_MODEL_ID", "modelo-de-teste")
    s = Settings(_env_file=None)
    assert s.llm_model_id == "modelo-de-teste"
