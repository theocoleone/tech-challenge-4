"""Modelos de embedding do projeto, com cache em disco.

- "titan": Amazon Titan Text Embeddings V2 no Bedrock (execução principal).
- "local": paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers), usado
  como comparação e como modo de reprodução sem conta AWS.

As duas classes seguem a interface de embeddings do LangChain
(embed_documents / embed_query), então servem direto para o Chroma.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.embeddings import Embeddings
from urllib3.exceptions import ReadTimeoutError

from src.config import check_aws_account, get_settings

MODELOS = {
    "titan": "amazon.titan-embed-text-v2:0",
    "local": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


class TitanEmbeddings(Embeddings):
    """Titan V2 via invoke_model. A API recebe um texto por chamada, então paraleliza."""

    def __init__(self, dimensoes: int = 1024, workers: int = 16):
        import boto3
        from botocore.config import Config

        s = get_settings()
        check_aws_account()
        self.model_id = MODELOS["titan"]
        self.dimensoes = dimensoes
        self.workers = workers
        self.client = boto3.Session(profile_name=s.aws_profile, region_name=s.aws_region).client(
            "bedrock-runtime",
            config=Config(
                retries={"mode": "adaptive", "max_attempts": 10},
                read_timeout=120,
                max_pool_connections=workers * 2,
            ),
        )

    def _um(self, texto: str, tentativas: int = 6) -> list[float]:
        corpo = json.dumps({"inputText": texto, "dimensions": self.dimensoes, "normalize": True})
        for tentativa in range(tentativas):
            try:
                resposta = self.client.invoke_model(modelId=self.model_id, body=corpo)
                return json.loads(resposta["body"].read())["embedding"]
            except (ReadTimeoutError, BotoCoreError, ClientError) as erro:
                # O retry do boto3 cobre a chamada, mas não a leitura do corpo da
                # resposta; um timeout ali derrubava o processo inteiro.
                if tentativa == tentativas - 1:
                    raise
                time.sleep(2 ** tentativa)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._um, textos))

    def embed_query(self, texto: str) -> list[float]:
        return self._um(texto)


class LocalEmbeddings(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model_id = MODELOS["local"]
        self.modelo = SentenceTransformer(self.model_id)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return self.modelo.encode(textos, batch_size=128, normalize_embeddings=True, show_progress_bar=True).tolist()

    def embed_query(self, texto: str) -> list[float]:
        return self.modelo.encode(texto, normalize_embeddings=True).tolist()


def get_embedder(nome: str) -> Embeddings:
    if nome == "titan":
        return TitanEmbeddings()
    if nome == "local":
        return LocalEmbeddings()
    raise ValueError(f"modelo de embedding desconhecido: {nome} (use {list(MODELOS)})")


def embeddings_com_cache(nome: str, ids: list[str], textos: list[str], cache_dir: Path, lote: int = 2000) -> np.ndarray:
    """Calcula os vetores só dos textos ainda não vistos e guarda tudo em parquet.

    A chave do cache é (review_id, texto): se o texto de uma avaliação mudar,
    o vetor é recalculado. Assim o índice pode ser refeito sem pagar de novo.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    arquivo = cache_dir / f"embeddings_{nome}.parquet"
    cache = pd.read_parquet(arquivo) if arquivo.exists() else pd.DataFrame(columns=["review_id", "texto", "vetor"])
    conhecidos = {(r, t): v for r, t, v in zip(cache.review_id, cache.texto, cache.vetor)}

    faltando = [(i, t) for i, t in zip(ids, textos) if (i, t) not in conhecidos]
    if faltando:
        embedder = get_embedder(nome)
        for inicio in range(0, len(faltando), lote):
            parte = faltando[inicio:inicio + lote]
            vetores = embedder.embed_documents([t for _, t in parte])
            for (i, t), v in zip(parte, vetores):
                conhecidos[(i, t)] = np.asarray(v, dtype=np.float32)
            pd.DataFrame(
                {"review_id": [k[0] for k in conhecidos], "texto": [k[1] for k in conhecidos], "vetor": list(conhecidos.values())}
            ).to_parquet(arquivo, index=False)  # salva a cada lote: uma queda não perde o que já foi pago
            print(f"  {nome}: {min(inicio + lote, len(faltando)):,}/{len(faltando):,} novos vetores")
    return np.vstack([np.asarray(conhecidos[(i, t)], dtype=np.float32) for i, t in zip(ids, textos)])
