""" Deepnlp Package """
from __future__ import unicode_literals

__version__ = '0.1.7'
__license__ = 'MIT'

from deepnlp import downloader
from deepnlp import model_util

# global function for download pre-trained model from github
# https://github.com/rockingdingo
download = downloader.download
register_model = model_util.register_model
