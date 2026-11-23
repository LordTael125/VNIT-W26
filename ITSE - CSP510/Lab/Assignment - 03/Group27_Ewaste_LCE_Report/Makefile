# Makefile for Group 27 — CSL510ISE W26 Practical 3 Report
TEX = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error
TARGET = main

.PHONY: all pdf clean distclean

all: pdf

pdf: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex config/*.tex frontmatter/*.tex figures/*.tex sections/*.tex backmatter/*.tex assets/*
	$(TEX) $(FLAGS) $(TARGET).tex
	$(TEX) $(FLAGS) $(TARGET).tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(TARGET).pdf

