# Instrucciones para Remover Archivo Grande del Historial de Git

El archivo `data/ml_dataset.csv` (117.19 MB) excede el límite de 100 MB de GitHub y está bloqueando el push.

## Solución Recomendada: Rebase Interactivo

### Paso 1: Iniciar Rebase Interactivo
```powershell
git rebase -i b45433b
```

### Paso 2: Editar el Archivo de Rebase
Cuando se abra el editor, cambia `pick` por `edit` en estos dos commits:
- `b8bf436 mas datos, y empezado a limpiarlos`
- `1c32355 mas datos, y empezado a limpiarlos`

Debería verse así:
```
edit b8bf436 mas datos, y empezado a limpiarlos
pick e6615d2 mas datos, y empezado a limpiarlos
pick 184bdae mas datos, y empezado a limpiarlos
pick 3b7d25d mas datos, y empezado a limpiarlos
pick 5250c04 mas datos, y empezado a limpiarlos
edit 1c32355 mas datos, y empezado a limpiarlos
...
```

Guarda y cierra el editor.

### Paso 3: Remover el Archivo del Primer Commit
Cuando Git se detenga en el commit `b8bf436`:
```powershell
git rm --cached data/ml_dataset.csv
git commit --amend --no-edit
git rebase --continue
```

### Paso 4: Remover el Archivo del Segundo Commit
Cuando Git se detenga en el commit `1c32355`:
```powershell
git rm --cached data/ml_dataset.csv
git commit --amend --no-edit
git rebase --continue
```

### Paso 5: Hacer Force Push
```powershell
git push --force-with-lease
```

## Alternativa: Usar BFG Repo-Cleaner (Más Fácil)

1. Descargar BFG: https://rtyley.github.io/bfg-repo-cleaner/
2. Ejecutar:
```powershell
java -jar bfg.jar --delete-files data/ml_dataset.csv
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force-with-lease
```

## Nota Importante

⚠️ **ADVERTENCIA**: Estos comandos reescriben el historial de Git. Si otros colaboradores tienen el repositorio, deben hacer:
```powershell
git fetch origin
git reset --hard origin/main
```
