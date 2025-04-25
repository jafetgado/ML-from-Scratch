#!/bin/bash

git status
git add .
git status
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "Auto-commit on $timestamp"
git push -u origin main
