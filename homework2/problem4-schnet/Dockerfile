# Define the base image
FROM --platform=linux/x86_64 ubuntu:22.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change the default shell
# Remark: '--login' ensures that both ~/.profile and ~/.bashrc are sourced properly
#         that is necessary in order to use various conda commands inside the image.
SHELL [ "/bin/bash", "--login", "-c" ]

# Define environment variables for a non-root user and create this user
ENV USER non_root_user
ENV UID 1000
ENV GID 100
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# Copy files to the image, change the owner, and set their permissions
# Remark: Do this while still being the root user
COPY scripts/install.sh /tmp/
RUN chown $UID:$GID /tmp/install.sh && \
    chmod u+x /tmp/install.sh

COPY scripts/entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

# Change to the created non-root user
USER $USER

# Install miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# Make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# Make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# Create a project directory inside user's home directory
ENV PROJECT_DIR $HOME/project
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# Build the conda environment
ENV ENV_NAME base
RUN conda update --name base --channel defaults conda && \
    conda clean --all --yes
#conda create --name $ENV_NAME python=3.10 && \

# Install the python packages
RUN conda activate $ENV_NAME && \
    /tmp/install.sh && \
    conda deactivate

# Expose the (container internal) port 8888
EXPOSE 8888

RUN cat /usr/local/bin/entrypoint.sh >> ~/.bashrc

# The code to run when container is started
#ENTRYPOINT [ "/usr/local/bin/entrypoint.sh", "bash" ]
