FROM tensorflow/tensorflow:2.6.0-gpu-jupyter
RUN pip install tfds-nightly 
RUN pip install fire

