#!/bin/bash

# Install distutils
sudo apt-get install -y python3-distutils

# Install Python dependencies from requirements.txt
pip install -r requirements.txt
