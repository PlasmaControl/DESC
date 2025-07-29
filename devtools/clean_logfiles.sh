#!/bin/sh

if [ -f "devtools/pre-commit.log" ]; then
   echo "deleting devtools/pre-commit.log file..."
   rm "devtools/pre-commit.log"
fi
