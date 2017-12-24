#!/bin/bash

LOCAL_PATH="$(cd `dirname $0`; pwd)"
# Download CRFPP 0.58
curl -o ${LOCAL_PATH}/CRF-0.58.tar.gz "http://www.deepnlp.org/downloads/?project=crfpp&file_path=/&file_name=CRF-0.58.tar.gz"
echo "Downloading CRF++ package to local directory"
echo ${LOCAL_PATH}/"CRF-0.58.tar.gz"

# Untar and install
echo "Start install crf++"
tar xzvf ${LOCAL_PATH}/CRF-0.58.tar.gz
cd CRF++-0.58
./configure
sudo make && sudo make install

# install python interface
cd python
python setup.py build
python setup.py install
sudo ln -s /usr/local/lib/libcrfpp.so.0 /usr/lib/
