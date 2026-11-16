PYTHON ?= python3.12
PY := .venv/bin/python

.PHONY: setup test smoke data notebooks index temas golden pool julgamentos eval-busca eval-filtros eval-golden calibrar api

setup:  ## cria o venv e instala as dependências
	test -d .venv || $(PYTHON) -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:   ## roda os testes
	$(PY) -m pytest -q

smoke:  ## testa a conexão com o Bedrock
	$(PY) -m src.smoke_bedrock

data:   ## monta a base de conhecimento (data/processed/)
	$(PY) -m src.data.build_documents

notebooks: ## reexecuta os notebooks salvando as saídas
	$(PY) -m src.run_notebook notebooks/01_base_conhecimento.ipynb
	$(PY) -m src.run_notebook notebooks/02_eda_reviews.ipynb
	$(PY) -m src.run_notebook notebooks/03_metadados.ipynb

index:  ## gera os índices BM25 e vetoriais (titan + local), com cache de embeddings
	$(PY) -m src.index.build_index

golden: ## confere os rótulos do golden set e da calibração contra a base
	$(PY) -m src.eval.check_golden

pool:   ## monta eval/julgamentos.xlsx para marcar relevância (preserva marcações)
	$(PY) -m src.eval.pool

julgamentos: ## copia as marcações do Excel para eval/julgamentos.csv (o que vai para o git)
	$(PY) -m src.eval.julgamentos

eval-busca: ## compara BM25, denso, híbrida e híbrida + rerank (eval/results/)
	$(PY) -m src.eval.eval_retrieval

temas:  ## agrupa as avaliações em temas (KMeans + rótulos do LLM)
	$(PY) -m src.analytics.themes

calibrar: ## calibra o limiar de sem evidência (eval/calibracao.jsonl)
	$(PY) -m src.eval.calibrar

eval-filtros: ## compara a recuperação com e sem filtros de metadados
	$(PY) -m src.eval.eval_filtros

eval-golden: ## roda o golden set de ponta a ponta (ramos, citações, latência)
	$(PY) -m src.eval.run_eval

api:    ## sobe a API e o front em http://127.0.0.1:8000 (só nesta máquina)
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
