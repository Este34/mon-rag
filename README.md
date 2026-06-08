# mon-rag — un mini système RAG, fait à la main

Un petit système **RAG** (*Retrieval-Augmented Generation*) en Python, 100 % local,
sans framework lourd. Il répond à des questions en s'appuyant sur **tes propres documents**.

> Projet d'apprentissage : le but est de comprendre chaque maillon (recherche →
> augmentation → génération). Le guide pas-à-pas complet est dans [`docs/guide.md`](docs/guide.md).

## Comment ça marche (en 3 temps)

1. **Recherche** (*Retrieval*) — on retrouve les passages les plus pertinents pour la question.
2. **Augmentation** — on colle ces passages dans le prompt, avant la question.
3. **Génération** — un LLM local (Ollama) répond en s'appuyant dessus.

## Installation

```powershell
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Il faut aussi **Ollama** (moteur LLM local) : télécharge-le sur https://ollama.com puis :

```powershell
ollama pull llama3.2
```

## Utilisation

```powershell
python rag.py
```

Pose une question, par ex. *« C'est quoi un embedding ? »*. Tape `quit` pour sortir.

## Structure

| Fichier / dossier | Rôle |
|-------------------|------|
| `rag.py` | le programme principal (recherche + génération) |
| `corpus/` | les documents source (un `.txt` = un document) |
| `docs/guide.md` | le guide d'apprentissage pas-à-pas |

> ⚠️ Les documents confidentiels vont dans `corpus/prive/` (ignoré par git, voir `.gitignore`).

## Briques techniques

- **sentence-transformers** (`all-MiniLM-L6-v2`) — texte → embeddings (vecteurs)
- **numpy** — similarité cosinus entre vecteurs
- **ollama** (`llama3.2`) — le LLM local qui génère la réponse
