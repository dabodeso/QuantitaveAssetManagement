# Script para remover archivo grande del historial de Git
# Ejecutar desde el directorio del repositorio

Write-Host "Removiendo data/ml_dataset.csv del historial de Git..." -ForegroundColor Yellow

# Encontrar el commit base (antes del primer commit con el archivo)
$baseCommit = "b45433b"

# Hacer rebase interactivo
Write-Host "Iniciando rebase interactivo..." -ForegroundColor Yellow
Write-Host "NOTA: Esto abrirá un editor. Debes editar los commits para remover el archivo." -ForegroundColor Yellow

# Alternativa: usar git filter-branch con sintaxis de Windows
Write-Host "`nUsando método alternativo con git filter-branch..." -ForegroundColor Yellow

# Remover el archivo del historial usando git filter-branch
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch data/ml_dataset.csv' --prune-empty --tag-name-filter cat -- --all

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nArchivo removido exitosamente del historial." -ForegroundColor Green
    Write-Host "Ahora ejecuta: git push --force-with-lease" -ForegroundColor Yellow
} else {
    Write-Host "`nError al remover el archivo. Intentando método alternativo..." -ForegroundColor Red
    
    # Método alternativo: rebase interactivo manual
    Write-Host "`nDebes hacer un rebase interactivo manual:" -ForegroundColor Yellow
    Write-Host "1. git rebase -i b45433b" -ForegroundColor Cyan
    Write-Host "2. Cambiar 'pick' a 'edit' en los commits b8bf436 y 1c32355" -ForegroundColor Cyan
    Write-Host "3. En cada commit, ejecutar: git rm --cached data/ml_dataset.csv" -ForegroundColor Cyan
    Write-Host "4. git commit --amend --no-edit" -ForegroundColor Cyan
    Write-Host "5. git rebase --continue" -ForegroundColor Cyan
}
