# Dicionário de dados

Descrição em linguagem de negócio dos arquivos gerados pelo pipeline. Os CSVs originais da Olist estão documentados na [página do dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data).

## `data/processed/reviews.parquet` — todas as avaliações

Uma linha por avaliação (98.410), **com e sem comentário**. É a tabela usada para estatísticas gerais de nota e metadados, porque 57% dos clientes só dão a nota. As contagens por assunto usam `documents.parquet` junto de `temas.parquet`.

## `data/processed/documents.parquet` — base de busca

As mesmas colunas, filtradas para as 42.121 avaliações com texto útil. É o que vai para o índice.

| Campo | Tipo | Significado | Origem |
|---|---|---|---|
| `review_id` | texto | Código da avaliação. É o identificador citado nas respostas | reviews |
| `order_ids` | texto | Código do pedido avaliado; mais de um quando a avaliação estava ligada a vários | reviews |
| `review_score` | 1 a 5 | Nota dada pelo cliente | reviews |
| `texto_original` | texto | Título e comentário como o cliente escreveu, com dados pessoais substituídos. É o texto mostrado como evidência | reviews |
| `texto_normalizado` | texto | Versão usada nas buscas: pontuação separada, abreviações expandidas, acentos e negações preservados | derivado |
| `tem_texto` | booleano | A avaliação tem título ou comentário | derivado |
| `tem_titulo` | booleano | O cliente preencheu o título | reviews |
| `tem_mensagem` | booleano | O cliente preencheu o comentário | reviews |
| `indexavel` | booleano | O texto tem conteúdo útil (exclui "10", "...", "A") | derivado |
| `n_palavras` | inteiro | Palavras do comentário, sem contar pontuação | derivado |
| `data_avaliacao` | aaaammdd | Dia em que o cliente respondeu a pesquisa | reviews |
| `data_envio_pesquisa` | aaaammdd | Dia em que a pesquisa foi enviada ao cliente | reviews |
| `ano_mes_compra` | aaaamm | Mês da compra. É o campo usado nos filtros de período | orders |
| `order_status` | texto | Situação do pedido: entregue, cancelado, enviado, indisponível, faturado, em processamento ou criado | orders |
| `valor_total` | reais | Preço mais frete de todos os itens do pedido | items |
| `multi_pedido` | booleano | A mesma avaliação estava ligada a mais de um pedido | derivado |
| `entregue` | booleano | O pedido chegou ao cliente | orders |
| `dias_atraso` | inteiro ou vazio | Dias entre a entrega real e a prevista; positivo significa atraso. Vazio quando o pedido não foi entregue | derivado |
| `entregue_com_atraso` | booleano | A entrega passou da data prevista | derivado |
| `categoria_pt` | texto | Categoria do item de maior valor do pedido, em português | products |
| `categoria_en` | texto | A mesma categoria em inglês | tradução |
| `multi_categoria` | booleano | O pedido tinha produtos de categorias diferentes | derivado |
| `n_itens` | inteiro | Quantidade de itens do pedido | items |
| `uf_cliente` | texto | Estado do cliente (sigla) | customers |

## `data/processed/temas.parquet` — tema de cada avaliação

Usada nas perguntas de frequência. Cobre as avaliações de nota 1, 2, 4 e 5 (a nota 3 fica fora, por ser mista).

| Campo | Tipo | Significado |
|---|---|---|
| `review_id` | texto | Código da avaliação |
| `grupo` | texto | `nota_baixa` (1 e 2) ou `nota_alta` (4 e 5) |
| `tema_id` | texto | Grupo do agrupamento automático |
| `tema` | texto | Chave do tema depois de unir grupos do mesmo assunto |
| `nome_tema` | texto | Nome exibido do tema ("atraso na entrega") |
| `macro` | texto | Macro-tema escolhido de uma taxonomia de sete opções. No catálogo atual aparecem entrega, produto, troca e devolução e compra em geral |
| `dist_centro` | decimal | Distância ao centro do grupo; menor significa mais típica do tema |

## `data/processed/build_stats.json`

Quantidades de cada etapa da preparação, usadas no README e nos notebooks.

## `eval/` — material de avaliação

| Arquivo | O que é |
|---|---|
| `golden_set.jsonl` | 44 perguntas com o comportamento esperado de cada uma. Suíte de regressão usada durante o desenvolvimento; não é um holdout independente |
| `calibracao.jsonl` | 40 perguntas (20 com evidência, 20 sem) usadas para escolher o limiar de recusa |
| `julgamentos.csv` | Pool de 1.117 pares pergunta × avaliação, todos rotulados como relevantes ou não (869 e 248) |
| `results/` | Resultados datados de cada medição e o catálogo de temas |
