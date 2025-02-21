# Imagem do PyTorch com suporte ao CUDA
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Define o diretório de trabalho
WORKDIR /ws

# Instala dependências do sistema e o git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Atualiza o pip e instala as dependências do projeto
RUN pip install --upgrade pip && \
    pip install bitsandbytes && \
    pip install git+https://github.com/huggingface/transformers.git && \
    pip install git+https://github.com/huggingface/peft.git && \
    pip install git+https://github.com/huggingface/accelerate.git && \
    pip install huggingface_hub

# Copiar codigo de aplicação para o container
COPY . /ws

RUN mkdir -p /ws/results
VOLUME ["/ws/results"]

CMD ["/bin/bash"]
