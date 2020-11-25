FROM ubuntu:18.04 as builder

################################################################################
# Build Kaldi using MKL installed with conda
################################################################################

# Dynamic parameters
ARG PYTHON_VERSION

# Constant parameters
ENV BASH_ENV=/opt/bash_env
ENV CONDA_DIR=/opt/conda

# 1. Install essentials
RUN apt update -q && apt install -y wget git build-essential && rm -rf /var/lib/apt/lists/*

# 2. Install conda
ADD ./scripts/setup_conda_env.sh /src/
RUN bash /src/setup_conda_env.sh

# 3. Build Kaldi
ARG KALDI_COMMIT=0c6a3dcf0
ENV KALDI_DIR=/opt/kaldi

ADD ./scripts/build_kaldi.sh /src/
RUN bash /src/build_kaldi.sh

# 4. Copy featbins and dependent libraries
# NOTE: MKL libraries are not copied, but the binaries are linked to the MKL in the conda env
# so the same MKL has to be installed via conda in the final image
ADD ./scripts/copy_kaldi_executables.sh /src/
RUN bash /src/copy_kaldi_executables.sh "${KALDI_DIR}" /kaldi

################################################################################
# Build the final image
################################################################################
FROM ubuntu:18.04

# Dynamic parameters
ARG PYTHON_VERSION

# Constant parameters
ENV BASH_ENV=/opt/bash_env
ENV CONDA_DIR=/opt/conda

# 1. Install the essentials
RUN apt update -q && apt install -y wget git build-essential && rm -rf /var/lib/apt/lists/*

# 2. Install conda
ADD ./scripts/setup_conda_env.sh /src/
RUN bash /src/setup_conda_env.sh

# 3. Add the copied libraries to LD_LIBRARY_PATH
COPY --from=builder /kaldi /opt/kaldi
ENV PATH="${PATH}:/opt/kaldi/bin" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/kaldi/lib"
