# Mon premier RAG — guide pas-à-pas (avec Claude comme tuteur)

**Objectif :** construire, sur ta machine, un vrai mini-système RAG qui répond à des questions à partir de tes propres documents.
**Durée :** 2 à 4 heures.
**À la fin tu auras :** un programme Python qui marche, et surtout la *compréhension* de chaque maillon pour le refaire seul et l’améliorer.

> On le fait **à la main**, sans gros framework (pas de LangChain pour l’instant). C’est volontaire : tu vas voir comment ça marche vraiment à l’intérieur. Les frameworks, ce sera l’étape d’après, une fois que tu auras compris.

-----

## Comment te servir de ce guide

À chaque étape : 1) lis l’explication, 2) tape le code toi-même (ne copie-colle pas, **réécris-le** — c’est comme ça qu’on apprend), 3) lance-le, 4) si tu bloques ou tu veux comprendre une ligne, utilise les prompts de la dernière section pour me demander.

-----

## Partie 0 — C’est quoi un RAG, en une image

RAG = *Retrieval-Augmented Generation* = « génération augmentée par la recherche ».

Le problème : un LLM ne connaît pas TES documents (ceux du CEA, par ex.). Si tu lui poses une question dessus, il invente.

La solution RAG, en 3 temps :

1. **Recherche** : parmi tes documents, on retrouve les passages les plus pertinents pour la question.
1. **Augmentation** : on colle ces passages dans le prompt, juste avant la question.
1. **Génération** : le LLM répond en s’appuyant sur ces passages.

C’est tout. Le reste, ce sont des détails techniques de ces 3 étapes.

-----

## Partie 1 — Installer l’environnement

### 1.1 Python et un dossier de projet

Crée un dossier `mon-rag`, ouvre un terminal dedans, puis crée un environnement virtuel (un espace isolé pour les librairies de ce projet) :

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
```

> **Pourquoi un venv ?** Pour que les librairies de ce projet n’aillent pas se mélanger avec celles des autres projets. Bonne habitude de pro.

### 1.2 Installer les librairies

```bash
pip install sentence-transformers numpy ollama
```

À quoi sert chacune :

- **`sentence-transformers`** : transforme du texte en *embeddings* (des vecteurs de nombres). C’est notre « traducteur texte → maths ».
- **`numpy`** : calcul sur des tableaux de nombres. On s’en sert pour mesurer la proximité entre vecteurs.
- **`ollama`** : la librairie Python pour parler à un LLM qui tourne en local sur ta machine.

### 1.3 Installer Ollama (le moteur du LLM)

Télécharge Ollama sur **ollama.com**, installe-le, puis récupère un petit modèle :

```bash
ollama pull llama3.2
```

> Tu peux aussi essayer `qwen2.5` (très bon et léger). Un « petit » modèle suffit largement pour apprendre, et ça tourne sans carte graphique de gamer.

**Checkpoint :** `ollama run llama3.2` doit te permettre de discuter avec le modèle dans le terminal. Tape `/bye` pour sortir.

-----

## Partie 2 — Le code, étape par étape

Crée un fichier `rag.py`. On le remplit petit à petit.

### Étape 1 — Le corpus

Pour démarrer simple, on met quelques documents directement dans le code (en vrai, on les lirait depuis des fichiers — on verra ça en Partie 4).

```python
documents = [
    "Le CEA de Marcoule est spécialisé dans la chimie séparative et le cycle du combustible.",
    "Un système RAG combine une recherche documentaire avec un modèle de langage.",
    "Les embeddings transforment du texte en vecteurs de nombres.",
    "La similarité cosinus mesure si deux vecteurs pointent dans la même direction.",
]
```

> **Idée clé :** chaque élément de cette liste est un « morceau » (*chunk*) que le système pourra retrouver. Plus tard, tu découperas tes vrais documents en morceaux de ce genre.

### Étape 2 — Transformer le texte en embeddings

Un *embedding*, c’est une liste de nombres qui représente le **sens** d’un texte. Deux textes au sens proche auront des vecteurs proches. C’est ce qui permet de chercher « par le sens » et pas juste « par mots-clés ».

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")   # téléchargé 1 fois, puis local
doc_embeddings = embedder.encode(documents)          # un vecteur par document
```

> `all-MiniLM-L6-v2` est un modèle d’embeddings petit et rapide. Chaque texte devient un vecteur de 384 nombres.

### Étape 3 — La recherche (le cœur du RAG)

Ici on **démystifie la « base vectorielle »** : pour quelques documents, c’est juste un calcul de proximité. On embed la question, puis on mesure la **similarité cosinus** entre la question et chaque document, et on garde les plus proches.

```python
import numpy as np

def rechercher(question, k=2):
    q_emb = embedder.encode([question])[0]
    sims = doc_embeddings @ q_emb / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = np.argsort(sims)[::-1][:k]   # les k indices les plus similaires
    return [documents[i] for i in top_idx]
```

> **Ce qui se passe :** `@` fait le produit scalaire ; on divise par les normes → c’est la similarité cosinus (entre -1 et 1, plus c’est haut plus c’est proche). `argsort(...)[::-1]` trie du plus proche au moins proche.

### Étape 4 — Construire le prompt augmenté + générer

On colle les passages trouvés dans le prompt, et on demande au LLM de répondre **uniquement** à partir de ça.

```python
import ollama

def repondre(question):
    contexte = "\n".join(rechercher(question))
    prompt = f"""Réponds à la question en t'appuyant uniquement sur le contexte ci-dessous.
Si la réponse n'y est pas, dis-le.

Contexte :
{contexte}

Question : {question}
Réponse :"""
    sortie = ollama.generate(model="llama3.2", prompt=prompt)
    return sortie["response"]
```

> **Le « R » et le « G » du RAG sont là :** `rechercher()` = Retrieval, le prompt avec contexte = Augmented, `ollama.generate` = Generation.

### Étape 5 — Une boucle pour poser des questions

```python
if __name__ == "__main__":
    while True:
        q = input("\nTa question (ou 'quit') : ")
        if q.lower() == "quit":
            break
        print(repondre(q))
```

-----

## Partie 3 — Lancer et vérifier

```bash
python rag.py
```

Pose par exemple : *« C’est quoi un embedding ? »* ou *« Le CEA de Marcoule fait quoi ? »*.

**Checkpoints pour savoir si ça marche :**

- Le programme démarre sans erreur → l’environnement est bon.
- Il répond en s’appuyant sur tes documents → la chaîne RAG fonctionne.
- Pose une question hors-sujet (« quelle est la capitale du Japon ? ») : idéalement il dit qu’il ne sait pas, car ce n’est pas dans le contexte. C’est le signe que la « cage » du RAG fonctionne.

-----

## Partie 4 — Améliorer (une fois que la base marche)

Fais-en UNE à la fois, pas tout d’un coup.

1. **Lire de vrais fichiers** plutôt que la liste codée en dur :

   ```python
   import os
   documents = []
   for nom in os.listdir("corpus"):
       if nom.endswith(".txt"):
           with open(os.path.join("corpus", nom), encoding="utf-8") as f:
               documents.append(f.read())
   ```
1. **Le découpage (chunking)** : un vrai document est trop long. Coupe-le en morceaux de ~500 caractères ou par paragraphe. La qualité du chunking, c’est *exactement* ton métier sur le projet du CEA.
1. **Une vraie base vectorielle** (Chroma ou FAISS) quand tu auras beaucoup de documents : même principe que ta fonction `rechercher`, mais optimisé pour des milliers de morceaux.
1. **Évaluer** : prépare 5 questions dont tu connais la bonne réponse, et vérifie que ton RAG les trouve. C’est ce qui sépare l’amateur du pro.

-----

## Partie 5 — Utiliser Claude comme tuteur

À chaque étape, copie-colle-moi une de ces demandes (en ajoutant ton code ou ton erreur) :

- **Comprendre une ligne :** « Explique-moi cette ligne mot par mot, comme si je débutais : `[colle la ligne]` »
- **Débloquer une erreur :** « J’ai cette erreur quand je lance `rag.py`, aide-moi à comprendre la cause et à corriger (sans me donner direct la solution, guide-moi) : `[colle l'erreur]` »
- **Vérifier que j’ai compris :** « Interroge-moi avec 3 questions pour vérifier que j’ai compris les embeddings et la similarité cosinus. »
- **Approfondir :** « Pourquoi on utilise la similarité cosinus et pas une autre mesure de distance ? »
- **Étape suivante :** « J’ai fini la version de base, fais-moi faire l’amélioration “chunking” en m’expliquant chaque choix. »
- **Relecture :** « Voici mon code complet, dis-moi ce qui est bien et ce que je pourrais améliorer, et pourquoi. »

> Conseil de tuteur : demande-moi de t’**expliquer** ou de te **guider**, pas juste de « donner le code ». C’est en cherchant un peu que ça rentre.

-----

## En résumé

Tu construis les 3 briques du RAG (rechercher → augmenter → générer), à la main, en local avec Ollama. Une fois que ça tourne et que tu comprends chaque ligne, tu sauras le refaire, l’améliorer, et surtout en parler avec assurance — y compris au CEA avec le dev de l’outil.

**Premier objectif réaliste :** ce week-end, avoir la version de base qui répond à une question sur tes 4 documents. Rien de plus. Le reste viendra ensuite.
