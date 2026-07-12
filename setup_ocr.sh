#!/bin/bash
# One-time local OCR setup for vision.py's local pipeline (no Gemini key
# needed -- see vision.py's module docstring for why this exists).
#
# 1. tesseract binary (not a pip package)
# 2. targeted language packs (tessdata_fast: compact, good enough for
#    printed menus/labels/receipts) -- the brew formula only ships eng/osd/snum
set -euo pipefail

brew install tesseract

TESSDATA="$(brew --prefix tesseract)/share/tessdata"
LANGS="jpn tha ita chi_sim chi_tra kor vie spa fra deu por ell"

for lang in $LANGS; do
  if [ ! -f "$TESSDATA/${lang}.traineddata" ]; then
    curl -sL -o "$TESSDATA/${lang}.traineddata" \
      "https://github.com/tesseract-ocr/tessdata_fast/raw/main/${lang}.traineddata" &
  fi
done
wait

echo "Tesseract + language packs ready: $(ls "$TESSDATA" | grep -c traineddata) traineddata files in $TESSDATA"
python3 -m pip install pytesseract
