import pytest

import src.config as config


class _FakeSTS:
    def __init__(self, conta):
        self.conta = conta

    def get_caller_identity(self):
        return {"Account": self.conta}


class _FakeSession:
    conta = None

    def __init__(self, profile_name=None, region_name=None):
        pass

    def client(self, nome):
        assert nome == "sts"
        return _FakeSTS(self.conta)


@pytest.fixture(autouse=True)
def _limpa_cache():
    config.check_aws_account.cache_clear()
    config.get_settings.cache_clear()
    yield
    config.check_aws_account.cache_clear()
    config.get_settings.cache_clear()


def _sessao_na_conta(monkeypatch, conta):
    import boto3

    _FakeSession.conta = conta
    monkeypatch.setattr(boto3, "Session", _FakeSession)


def test_conta_do_curso_passa(monkeypatch):
    monkeypatch.setenv("TC4_AWS_ACCOUNT_ID", "111111111111")
    _sessao_na_conta(monkeypatch, "111111111111")
    assert config.check_aws_account() == "111111111111"


def test_conta_de_trabalho_aborta(monkeypatch):
    monkeypatch.setenv("TC4_AWS_ACCOUNT_ID", "111111111111")
    _sessao_na_conta(monkeypatch, "222222222222")
    with pytest.raises(config.ContaAwsErrada):
        config.check_aws_account()


def test_get_llm_nao_cria_cliente_na_conta_errada(monkeypatch):
    monkeypatch.setenv("TC4_AWS_ACCOUNT_ID", "111111111111")
    _sessao_na_conta(monkeypatch, "222222222222")
    with pytest.raises(config.ContaAwsErrada):
        config.get_llm()


def test_conta_esperada_precisa_ser_configurada(monkeypatch):
    monkeypatch.delenv("TC4_AWS_ACCOUNT_ID", raising=False)
    # O teste precisa ser hermético: um .env local válido não pode transformar
    # este caso unitário numa chamada real ao STS.
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(_env_file=None))
    with pytest.raises(config.ContaAwsNaoConfigurada):
        config.check_aws_account()
