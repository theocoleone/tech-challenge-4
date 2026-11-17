"""Exporta as respostas reais do conjunto de avaliação para a demo estática.

A demo em docs/demo/ mostra as respostas que o sistema produziu de verdade, com
evidências, contagens e os casos de recusa. Ela não chama o Bedrock: serve para
quem quer explorar o resultado sem credencial da AWS e sem subir o serviço.

Uso: python -m src.eval.exportar_demo
"""
import json
from pathlib import Path

from src.config import ROOT
from src.eval.check_golden import carregar

TIPOS = {
    "especifica": "Perguntas específicas",
    "filtro": "Perguntas com recorte de metadados",
    "agregada": "Perguntas de frequência",
    "insuficiente": "Evidência insuficiente",
    "filtro_vazio": "Recorte sem nenhuma avaliação",
    "sem_evidencia": "Assunto que não existe na base",
    "fora_escopo": "Fora do escopo",
}


def ultimo_resultado() -> Path:
    arquivos = sorted((ROOT / "eval" / "results").glob("*_golden_ramos.json"))
    principais = [a for a in arquivos if a.name.count("_") == 3]  # ignora variantes "_pre_*"
    return (principais or arquivos)[-1]


def main() -> None:
    arquivo = ultimo_resultado()
    dados = json.loads(arquivo.read_text())
    esperado = {p["id"]: p for p in carregar(ROOT / "eval" / "golden_set.jsonl")}

    perguntas = []
    for r in dados["respostas"]:
        p = esperado.get(r["id"], {})
        perguntas.append({
            "id": r["id"],
            "grupo": TIPOS.get(p.get("tipo", ""), "Outras"),
            "pergunta": r["pergunta"],
            "status": r["status"],
            "resposta": r["resposta"],
            "insights": r["insights"],
            "evidencias": r["evidencias"],
            "contagem": r.get("contagem"),
            "filtros_aplicados": r["filtros_aplicados"],
            "pergunta_reescrita": r["pergunta_reescrita"],
            "n_no_recorte": r["n_no_recorte"],
            "tipo": r["tipo"],
            "tempos": r["tempos"],
        })

    resumo = {
        "gerado_em": arquivo.stem.split("_")[0],
        "n_perguntas": len(perguntas),
        "ramos_corretos": sum(x["acertou"] for x in dados["por_pergunta"]),
        "fidelidade": dados["geracao"]["fidelidade (afirmações sustentadas)"],
        "citacoes_validas": dados["geracao"]["citações válidas"],
        "exatidao_numerica": dados["agregadas"]["exatidão numérica (sem número fora da contagem)"],
        "modelos": {"embedding": "amazon.titan-embed-text-v2:0", "geracao": "us.amazon.nova-pro-v1:0"},
    }

    saida = ROOT / "docs" / "demo" / "respostas.json"
    saida.write_text(json.dumps({"resumo": resumo, "perguntas": perguntas}, ensure_ascii=False, indent=1))
    print(f"{len(perguntas)} respostas de {arquivo.name} → {saida.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
