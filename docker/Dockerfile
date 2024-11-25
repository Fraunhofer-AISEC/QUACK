FROM pytorch/pytorch:latest

WORKDIR /app

# Create virtual environment and activate it
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install additional packages
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install pennylane
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install torch

# Start Bash when container launches
CMD ["/bin/bash"]
