"""mon-rag : un mini systeme RAG fait a la main.

RAG = Retrieval (rechercher) -> Augmented (prompt + contexte) -> Generation (LLM).
Voir docs/guide.md pour les explications detaillees, etape par etape.
"""

import os

CORPUS_DIR = "corpus"


def charger_documents(dossier: str = CORPUS_DIR) -> list[str]:
    """Lit chaque fichier .txt du dossier et renvoie la liste des textes (un par fichier)."""
    documents: list[str] = []
    for nom in sorted(os.listdir(dossier)):
        if nom.endswith(".txt"):
            chemin = os.path.join(dossier, nom)
            with open(chemin, encoding="utf-8") as f:
                documents.append(f.read().strip())
    return documents


def rechercher(question: str, k: int = 2) -> list[str]:
    """Retrieval : renvoie les k passages les plus proches de la question.

    TODO (etape 3 du guide) :
      1. transformer la question en embedding,
      2. calculer la similarite cosinus avec chaque document,
      3. renvoyer les k documents les plus proches.
    """
    raise NotImplementedError("A implementer : voir docs/guide.md, etape 3")


def repondre(question: str) -> str:
    """Augmented + Generation : construit le prompt avec le contexte et appelle le LLM.

    TODO (etape 4 du guide) :
      1. recuperer le contexte via rechercher(question),
      2. construire un prompt qui demande de repondre UNIQUEMENT a partir du contexte,
      3. appeler ollama.generate(...) et renvoyer la reponse.
    """
    raise NotImplementedError("A implementer : voir docs/guide.md, etape 4")


def main() -> None:
    """Boucle interactive : pose des questions jusqu'a 'quit'."""
    while True:
        q = input("\nTa question (ou 'quit') : ")
        if q.lower() == "quit":
            break
        print(repondre(q))


if __name__ == "__main__":
    main()
