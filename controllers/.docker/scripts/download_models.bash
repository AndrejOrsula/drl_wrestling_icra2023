#!/usr/bin/env bash
set -e

## Download models
pip3 install -qqq --no-cache-dir gdown
mkdir -p /usr/local/webots-project/controllers/participant/models
gdown -q --no-cookies --remaining-ok --folder 1q36Jxg6F452Q3f7zvJPzaM3MKBD1Gps7 -O /usr/local/webots-project/controllers/participant/models
