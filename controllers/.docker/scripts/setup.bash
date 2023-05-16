#!/usr/bin/env bash
set -e

## Install Python dependencies
pip3 install -qqq --no-cache-dir ahrs opencv-python scipy
pip3 install -qqq --no-cache-dir "dreamerv3 @ git+https://github.com/danijar/dreamerv3.git"
pip3 install -qqq --no-cache-dir --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
wget -q https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/configs.yaml -O "$(pip show dreamerv3 | grep Location: | cut -d' ' -f2)/dreamerv3/configs.yaml"

## Download models
pip3 install -qqq --no-cache-dir gdown
mkdir -p /usr/local/webots-project/controllers/participant/models
gdown -q 15gZJPC3yW-vfb8s4dJfxkPJR8U8NFEtA -O /usr/local/webots-project/controllers/participant/models/model01.ckpt
gdown -q 1oe9QKsJd3RdrjBtJlmTW84PKEVxa5irs -O /usr/local/webots-project/controllers/participant/models/model02.ckpt
