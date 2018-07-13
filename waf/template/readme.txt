This directory holds the 8 components of the SVProc tool package:

(1) a subdirectory "styles" with
- the document class "svproc.cls",
- the Springer Nature MakeIndex style file "svind.ist"
- style file "svindd.ist", same as above - German version
- additional packages "required" by the SVProc-class
  (just in case they are missing from your installation):
  aliascnt.sty (part of the Oberdiek bundle)
  remreset.sty (by David Carlisle, needed bei 'aliascnt.sty')
Caveat Author: don't overwrite your actual versions of these
packages if they are already installed on your system

- a sub(sub)directory "bibtex" containing BibTeX styles

(2) a subdirectory "templates" with
- the sample text file "author.tex",
- the sample figure file "figure.eps" or "figure.pdf"
with preset class options, packages and coding examples;

Tip: Copy all these files to your working directory, run LaTeX
and produce your own example *.dvi or *.pdf file; rename the template
files as you see fit and use them for your own input.

(3) the pdf file "quickstart.pdf" with "essentials" for
an easy implementation of the "svproc" package

(4) the pdf file "authinst.pdf" with style and
coding instructions specific to -- Proceedings --

Tip: Follow these instructions to set up your files, to type in your
text and to obtain a consistent formal style; use these pages as
checklists before you submit your ready-to-print manuscript.

(5) the pdf file "refguide.pdf" describing the
SVProc document class features independent of any specific style
requirements.

Tip: Use it as a reference if you need to alter or enhance the
default settings of the SVProc document class and/or the templates.

(6) the pdf file "authsamp.pdf" as an example of a "contribution"

(7) a subdirectory "editor" containing

    - the sample root file "editor.tex"
      with preset class options, packages and coding examples;

Tip: Copy this file to your working directory,
run LaTeX on "editor.tex" and produce your own example file;
rename the template files as you see fit and
use them for your own input.

    - the pdf file "edsamp.pdf" as an example of a "complete book"

    - the pdf file "edinst.pdf" with step-by-step
      instructions for compiling all contributions to a single book.

(8) a file "history.txt" with version history
