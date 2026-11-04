PY := .venv/bin/python

.PHONY: setup test smoke

setup:  ## cria o venv e instala as dependências
	test -d .venv || ~/.local/bin/python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:   ## roda os testes
	$(PY) -m pytest -q

smoke:  ## testa a conexão com o Bedrock
	$(PY) -m src.smoke_bedrock
