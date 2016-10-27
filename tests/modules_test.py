#coding:utf-8
#Example for Sentences Segmentation

import sys,os
import codecs
import deepnlp.segmenter as segmenter
import deepnlp.pos_tagger as pos_tagger
import deepnlp.ner_tagger as ner_tagger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# concatenate tuples into one string "w1/t1 w2/t2 ..."
def _concat_tuples(tagging):
  TOKEN_BLANK = " "
  wl = [] # wordlist
  for (x, y) in tagging:
    wl.append(str(x + "/" + y))
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
pos_tagging = _concat_tuples(pos_tagger.predict(words))
ner_tagging = _concat_tuples(ner_tagger.predict(words))

fileOut.writelines(" ".join(words) + "\n")
fileOut.writelines(pos_tagging + "\n")
fileOut.writelines(ner_tagging + "\n")
fileOut.close

print (" ".join(words).encode('utf-8'))
print (pos_tagging.encode('utf-8'))
print (ner_tagging.encode('utf-8'))
