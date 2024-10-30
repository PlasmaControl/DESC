#!/bin/sh

if [ -f "devtools/pre-commit.log" ]; then
   echo "delete devtools/pre-commit.log"
   rm "devtools/pre-commit.log"
fi
