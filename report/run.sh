/Applications/LilyPond.app/Contents/Resources/bin/lilypond-book report.tex --pdf -o out
cd out
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
