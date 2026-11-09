"""Executa um notebook a partir da raiz do repo e salva as saídas nele.

Uso: python -m src.run_notebook notebooks/01_eda_reviews.ipynb
"""
import sys

import nbformat
from nbclient import NotebookClient


def main(caminho: str) -> None:
    nb = nbformat.read(caminho, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbformat.write(nb, caminho)
    print(f"executado: {caminho}")


if __name__ == "__main__":
    main(sys.argv[1])
