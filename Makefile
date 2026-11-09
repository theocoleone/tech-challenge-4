PY := .venv/bin/python

.PHONY: setup test smoke data notebooks index golden pool

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

index:  ## gera os índices BM25 e Chroma (titan + local), com cache de embeddings
	$(PY) -m src.index.build_index

golden: ## confere os rótulos do golden set e da calibração contra a base
	$(PY) -m src.eval.check_golden

pool:   ## monta eval/julgamentos.csv para marcar relevância (preserva marcações)
	$(PY) -m src.eval.pool
