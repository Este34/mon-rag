# Workflow de développement — antisèche

Bonnes habitudes git/GitHub pour ce projet.

## Le cycle

```
brancher  →  modifier  →  commiter  →  pousser  →  Pull Request  →  fusionner
```

```bash
git checkout main && git pull
git checkout -b feat/mon-sujet
# ... modifications ...
git add -A
git commit -m "Implemente la recherche par similarite cosinus"
git push -u origin feat/mon-sujet
# puis sur GitHub : ouvrir une Pull Request
```

## Préfixes de branche

| Préfixe | Pour quoi |
|---------|-----------|
| `feat/` | nouvelle fonctionnalité |
| `fix/`  | correction de bug |
| `docs/` | documentation |

## Filets de sécurité

- **pre-commit** : Ruff avant chaque commit (`python -m pre_commit install` une fois).
- **CI GitHub Actions** : revérifie à chaque push.

> 💡 Avec Claude : dis « sauvegarde sur git » pour un commit + push automatique.
