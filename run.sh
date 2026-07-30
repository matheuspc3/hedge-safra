#!/bin/bash
set -e
cd /home/mathe/Projects/desafio-itau
echo "=== 1. Backtest ==="
python3 scripts/backtest_hedge.py 2>&1
echo "=== 2. Logo ==="
python3 scripts/gerar_logo.py 2>&1
echo "=== 3. Graficos backtest ==="
python3 scripts/graficos_backtest.py 2>&1
echo "=== 4. Compilar apresentacao ==="
cd relatorio && pdflatex -interaction=nonstopmode apresentacao.tex 2>&1 | tail -5
pdflatex -interaction=nonstopmode apresentacao.tex 2>&1 | tail -5
echo "=== DONE ==="
