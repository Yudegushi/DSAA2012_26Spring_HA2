# Report Build Guide

This folder contains the LaTeX source for the HA2 report.

## Files

- `main.tex`: full report source.

## Compile

If LaTeX is installed and available in your PATH:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Output PDF:

- `report/main.pdf`

If your machine does not have a TeX distribution yet, install one first:

- Windows (recommended): MiKTeX or TeX Live (global install)
