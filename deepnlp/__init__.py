""" Deepnlp Package """
from __future__ import unicode_literals

__version__ = '0.1.6'
__license__ = 'MIT'

from deepnlp import downloader

# global function for download pre-trained model from github
# https://github.com/rockingdingo
download = downloader.download

