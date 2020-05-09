#!/bin/bash

cd dataset; mkdir glove
cd glove

ZIPFILE=glove.42B.300d.zip

echo "==> Downloading Common Crawl glove vectors..."
wget http://nlp.stanford.edu/data/$ZIPFILE

echo "==> Unzipping glove vectors..."
unzip $ZIPFILE
rm $ZIPFILE

echo "==> Done."

