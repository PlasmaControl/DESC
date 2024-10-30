#!/bin/sh

if [ -f "pre-commit.log" ]; then
   echo "delete pre-commit.log"
   rm "pre-commit.log"
fi
