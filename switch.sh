#!/bin/sh

mv app.py tmp
mv app2.py app.py
mv tmp app2.py

mv templates/index.html templates/tmp
mv templates/index2.html templates/index.html
mv templates/tmp templates/index2.html
