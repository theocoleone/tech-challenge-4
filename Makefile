PY := .venv/bin/python

.PHONY: setup test smoke data notebooks

setup:  ## cria o venv e instala as dependências
	test -d .venv || ~/.local/bin/python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:   ## roda os testes
	$(PY) -m pytest -q

smoke:  ## testa a conexão com o Bedrock
	$(PY) -m src.smoke_bedrock

data:   ## monta a base de conhecimento (data/processed/)
	$(PY) -m src.data.build_documents

notebooks: ## reexecuta os notebooks do F1 salvando as saídas
	$(PY) -m src.run_notebook notebooks/01_base_conhecimento.ipynb
	$(PY) -m src.run_notebook notebooks/02_eda_reviews.ipynb
