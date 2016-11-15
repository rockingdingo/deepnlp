#coding:utf-8
from __future__ import unicode_literals

import sys,os
import codecs

from deepnlp import segmenter
from deepnlp import pos_tagger # module: pos_tagger
from deepnlp import ner_tagger # module: ner_tagger

# Create new tagger instance
tagger_pos = pos_tagger.load_model(lang = 'zh')
tagger_ner = ner_tagger.load_model(lang = 'zh')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# concatenate tuples into one string "w1/t1 w2/t2 ..."
def _concat_tuples(tagging):
  TOKEN_BLANK = " "
  wl = [] # wordlist
  for (x, y) in tagging:
    wl.append(x + "/" + y)
  concat_str = TOKEN_BLANK.join(wl)
  return concat_str

# read input file
docs = []
file = codecs.open(os.path.join(BASE_DIR, 'docs_test.txt'), 'r', encoding='utf-8')
for line in file:
    line = line.replace("\n", "").replace("\r", "")
    docs.append(line)

# Test each individual module
# output file
fileOut = codecs.open(os.path.join(BASE_DIR, 'modules_test_results.txt'), 'w', encoding='utf-8')
words = segmenter.seg(docs[0])
pos_tagging = _concat_tuples(tagger_pos.predict(words))
ner_tagging = _concat_tuples(tagger_ner.predict(words))

fileOut.writelines(" ".join(words) + "\n")
fileOut.writelines(pos_tagging + "\n")
fileOut.writelines(ner_tagging + "\n")
fileOut.close

print (" ".join(words).encode('utf-8'))
print (pos_tagging.encode('utf-8'))
print (ner_tagging.encode('utf-8'))
