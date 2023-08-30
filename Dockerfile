FROM huggingface/transformers-pytorch-gpu:4.29.2
ADD requirements.txt /ws/requirements.txt
RUN pip install -r /ws/requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"