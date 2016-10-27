#!/usr/bin/env python
#coding:utf-8
# B, M, E, S: Beginning, Middle, End, Single

import codecs
import sys,os

import CRFPP

def crf_segmenter(input_file, output_file, tagger):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        tagger.clear()
        for word in line.strip():
            word = word.strip()
            if word:
                tagger.add((word + "\to\tB").encode('utf-8'))
        tagger.parse()
        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                char = tagger.x(i, j).decode('utf-8')
                tag = tagger.y2(i)
                if tag == 'B':
                    output_data.write(' ' + char)
                elif tag == 'M':
                    output_data.write(char)
                elif tag == 'E':
                    output_data.write(char + ' ')
                else: # tag == 'S'
                    output_data.write(' ' + char + ' ')
        output_data.write('\n')
    input_data.close()
    output_data.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "pls use: python crf_segmenter.py model input output"
        sys.exit()
    crf_model = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    tagger = CRFPP.Tagger("-m " + crf_model)
    crf_segmenter(input_file, output_file, tagger)
