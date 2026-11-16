"""API do Voice of Customer: pergunta em linguagem natural, resposta com evidências.

Serviço local, sem autenticação: a base tem texto escrito por clientes e cada
pergunta gasta chamadas pagas ao Bedrock. Por isso duas travas:

1. `make api` sobe em 127.0.0.1, então o serviço não é alcançável de outra máquina.
2. Toda requisição que não venha do próprio computador é recusada (middleware
   abaixo), caso alguém suba o servidor com outro endereço. Para usar na rede
   seria preciso autenticar de verdade; a variável TC4_PERMITIR_ACESSO_EXTERNO
   existe apenas para quem assumir esse risco conscientemente.

Uso: make api   (ou uvicorn src.api.main:app --host 127.0.0.1 --port 8000)
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from src.analytics.aggregate import Agregador
from src.config import get_settings
from src.generation.chain import K_CANDIDATOS, N_CONTEXTO, _buscador, responder
from src.generation.schemas import FiltroReviews
from src.retrieval.filters import mascara
from src.retrieval.rerank import reordenar

WEB = Path(__file__).resolve().parent.parent.parent / "web"
LOCAIS = {"127.0.0.1", "::1", "localhost", "testclient"}
app = FastAPI(title="Voice of Customer Intelligence", description="Perguntas sobre avaliações de clientes (Olist)")


@app.middleware("http")
async def so_local(request: Request, call_next):
    """Recusa o que não vem desta máquina, mesmo se o servidor subir com outro endereço."""
    import os

    cliente = request.client.host if request.client else ""
    if cliente not in LOCAIS and os.getenv("TC4_PERMITIR_ACESSO_EXTERNO") != "1":
        return JSONResponse({"detail": "serviço local: acesso permitido apenas do próprio computador"}, status_code=403)
    return await call_next(request)


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3, examples=["O que os clientes reclamam da entrega?"])
    filtros: FiltroReviews | None = Field(
        None, description="sobrescreve apenas os campos informados entre os filtros extraídos da pergunta"
    )


def _validar_categoria(filtros: dict) -> None:
    categoria = filtros.get("categoria_pt")
    if categoria and categoria not in set(_documentos().categoria_pt.unique()):
        raise HTTPException(422, f"categoria desconhecida: {categoria}")


@app.post("/ask", summary="Pergunta com resposta fundamentada nas avaliações")
def ask(p: Pergunta) -> dict:
    filtros = p.filtros.como_dict() if p.filtros else {}
    _validar_categoria(filtros)
    return responder(p.pergunta, filtros=filtros or None)


@app.post("/search", summary="Só a recuperação (para inspecionar o ranking)")
def search(p: Pergunta) -> dict:
    filtros = p.filtros.como_dict() if p.filtros else {}
    _validar_categoria(filtros)
    b = _buscador()
    candidatos = b.hibrida(p.pergunta, K_CANDIDATOS, filtros)
    pares = [(r.review_id, b.por_id.texto_original[r.review_id]) for r in candidatos]
    ordenados = reordenar(p.pergunta, pares)
    por_rrf = {r.review_id: r for r in candidatos}
    return {"pergunta": p.pergunta, "filtros": filtros, "n_no_recorte": int(mascara(b.docs, filtros).sum()),
            "resultados": [{"review_id": rid, "score_rerank": round(sc, 3),
                            "score_rrf": round(por_rrf[rid].score, 4), "posicoes": por_rrf[rid].origem,
                            "texto": b.por_id.texto_original[rid], "nota": int(b.por_id.review_score[rid]),
                            "categoria": b.por_id.categoria_pt[rid]}
                           for rid, sc in ordenados[:N_CONTEXTO * 2]]}


@app.get("/reviews/{review_id}", summary="Uma avaliação com todos os metadados")
def review(review_id: str) -> dict:
    por_id = _documentos().set_index("review_id")
    if review_id not in por_id.index:
        raise HTTPException(404, "avaliação não encontrada (ou sem texto, portanto fora do índice)")

    d = por_id.loc[review_id]
    saida = {"review_id": review_id}
    for col in por_id.columns:
        v = d[col]
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            saida[col] = None
        else:
            saida[col] = v.item() if hasattr(v, "item") else v
    return saida


@app.get("/themes", summary="Temas com contagem sobre todas as avaliações temáticas do recorte")
def themes(polaridade: str = Query("nota_baixa", pattern="^(nota_baixa|nota_alta)$"),
           review_score_min: int | None = Query(None, ge=1, le=5),
           review_score_max: int | None = Query(None, ge=1, le=5),
           categoria_pt: str | None = None, ano_mes_compra_min: int | None = None,
           ano_mes_compra_max: int | None = None, entregue_com_atraso: bool | None = None,
           order_status: str | None = None, uf_cliente: str | None = None,
           valor_min: float | None = Query(None, ge=0), valor_max: float | None = Query(None, ge=0),
           top: int = Query(10, ge=1, le=50)) -> dict:
    brutos = {k: v for k, v in locals().items() if k not in {"polaridade", "top"} and v is not None}
    try:
        filtros = FiltroReviews.model_validate(brutos).como_dict()
    except ValidationError as erro:
        raise HTTPException(
            422, detail=erro.errors(include_url=False, include_context=False)
        ) from erro
    _validar_categoria(filtros)
    if polaridade == "nota_baixa" and filtros.get("review_score_min", 1) > 2:
        raise HTTPException(422, "polaridade nota_baixa é incompatível com nota mínima acima de 2")
    if polaridade == "nota_alta" and filtros.get("review_score_max", 5) < 4:
        raise HTTPException(422, "polaridade nota_alta é incompatível com nota máxima abaixo de 4")
    ag = _agregador()
    pergunta = "elogios" if polaridade == "nota_alta" else "problemas"
    c = ag.contar(pergunta, filtros, None, top=top)
    return {"polaridade": c.polaridade, "filtros": filtros, "total_no_recorte": c.total_no_recorte,
            "total_com_tema": c.total_com_tema, "macros": c.macros, "temas": c.temas}


@app.get("/health", summary="Estado do serviço")
def health() -> dict:
    s = get_settings()
    docs = _documentos()
    return {"status": "ok", "documentos": len(docs), "temas": int(_agregador().temas.tema.nunique()),
            "modelos": {"analise": s.llm_model_id, "geracao": s.gen_model_id, "embedding": s.emb_model_id,
                        "provedor": s.llm_provider, "regiao": s.aws_region}}


@app.get("/categorias", summary="Categorias disponíveis para filtrar")
def categorias() -> list[str]:
    return sorted(_documentos().categoria_pt.unique())


@lru_cache
def _documentos() -> pd.DataFrame:
    """Metadados locais usados por endpoints que não precisam consultar o Bedrock."""
    return pd.read_parquet(get_settings().data_processed / "documents.parquet")


_agregador_cache: list[Agregador] = []


def _agregador() -> Agregador:
    if not _agregador_cache:
        _agregador_cache.append(Agregador())
    return _agregador_cache[0]


if WEB.exists():
    app.mount("/web", StaticFiles(directory=WEB), name="web")

    @app.get("/", include_in_schema=False)
    def raiz() -> FileResponse:
        return FileResponse(WEB / "index.html")
