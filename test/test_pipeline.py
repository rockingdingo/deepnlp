#coding:utf-8
#Set Default codec coding to utf-8 to print chinese correctly
import sys,os
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

import deepnlp.pipeline as p

# concatenate tuples into one string "w1/t1 w2/t2 ..."
def _concat_tuples(tagging):
  TOKEN_BLANK = " "
  wl = [] # wordlist
  for (x, y) in tagging:
    wl.append(str(x + "/" + y))
  concat_str = TOKEN_BLANK.join(wl)
  return concat_str

# input file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
docs = []
file = codecs.open(os.path.join(BASE_DIR, 'docs_test.txt'), 'r', encoding='utf-8')
for line in file:
    line = line.replace("\n", "").replace("\r", "")
    docs.append(line)

# output file
fileOut = codecs.open(os.path.join(BASE_DIR, 'pipeline_test_results.txt'), 'w', encoding='utf-8')

# analyze function
# @return: list of 3 elements [seg, pos, ner]
text = docs[0]
res = p.analyze(text)
words = p.segment(text)
pos_tagging = p.tag_pos(words)
ner_tagging = p.tag_ner(words)

# print pipeline.analyze() results
fileOut.writelines("pipeline.analyze results:" + "\n")
fileOut.writelines(res[0] + "\n")
fileOut.writelines(res[1] + "\n")
fileOut.writelines(res[2] + "\n")

print (res[0].encode('utf-8'))
print (res[1].encode('utf-8'))
print (res[2].encode('utf-8'))

# print modules results
fileOut.writelines("modules results:" + "\n")
fileOut.writelines(" ".join(words) + "\n")
fileOut.writelines(_concat_tuples(pos_tagging) + "\n")
fileOut.writelines(_concat_tuples(ner_tagging) + "\n")
fileOut.close
