/Applications/LilyPond.app/Contents/Resources/bin/lilypond-book report.tex --pdf -o lily
cd lily
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
