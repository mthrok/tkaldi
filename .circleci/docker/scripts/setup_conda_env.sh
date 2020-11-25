#!/usr/bin/evn bash

# Install conda and setup environment
# Write commands to BASH_ENV so that the environment is activated automatically

set -ex

wget --quiet -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash ./miniconda.sh -b -f -p "${CONDA_DIR}"
eval "$("${CONDA_DIR}"/bin/conda shell.bash hook)"
conda update --quiet -y conda
conda install --quiet -y python="${PYTHON_VERSION}"

printf 'eval "$(%s/bin/conda shell.bash hook)"\n' "${CONDA_DIR}" >> "${BASH_ENV}"
