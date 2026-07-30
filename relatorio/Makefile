# Compilação do relatório LaTeX
# Requer: texlive-core + mathpazo (Palatino)
# Compilar: make ou pdflatex → pdflatex (duas passagens para referências)

TEX = main.tex
AUX = main
PDF = main.pdf

.PHONY: all clean view

all: $(PDF)

$(PDF): $(TEX)
	pdflatex $(AUX)
	pdflatex $(AUX)

view: $(PDF)
	xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || start $(PDF)

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.toc
