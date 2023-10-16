# Use an official Python runtime as a parent image
#FROM python:3.10.12

# Using the Ubuntu image
FROM ubuntu:latest

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update package manager (apt-get) 
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt update

# installing python3 with a specific version
RUN apt install python3.10 -y
RUN apt install python3.10-distutils -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# installing other libraries
RUN apt-get install python3-pip -y && apt-get -y install sudo
RUN apt-get install python3-dev -y
RUN apt-get install curl -y
RUN apt-get install gnupg -y
RUN apt-get install nano -y
RUN apt-get install ca-certificates -y
RUN apt-get update && apt-get install -y git

# Create directories and import GPG key
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

# Add the Node.js repository
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list

# Install Node.js
RUN apt-get update && apt-get install nodejs -y

# Install tweet-harvest@latest
RUN npm install -g tweet-harvest@latest
RUN npm install -g npm@latest

WORKDIR /code

# install dependencies
COPY ./requirements.txt /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade --user -r /code/requirements.txt

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Check version
RUN python3 --version
RUN node -v

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "app.py"]
