#!/usr/bin/env python3
#
# preprocess/hkcancor.py
#
# This file preprocesses the XML-like hkcancor dataset into a tab-separated format
# with two columns: a binary (i.e. 0 or 1) label and a list of characters.  The input
# format is a set of files, each containing data in this structure:
#       <info> ... </info>
#       <sent>
#           ...
#           <sent_tag>
#               ... text ...
#           </sent_tag>
#           ...
#       </sent>
#       ...
#       <sent>
#           ...
#           <sent_tag>
#               ... text ...
#           </sent_tag>
#           ...
#       </sent>

import argparse
import itertools
import random

from xml.etree import ElementTree

# NOTE: "fp" stands for "file-pointer", since these objects point to the data inside
#       the file, rather than the filename.  This is because argparse.FileType does
#       the equivalent of `arg = open(filename)`.
def preprocess(input_fps, output_fp, zero_proportion):
    for input_fp in input_fps:
        # these XML files don't have a "root" (outermost) node, so we can add one :)
        contents = itertools.chain('<root>', input_fp, '</root>')
        root = ElementTree.fromstringlist(contents)
        # assuming that the input has the structure specified above, this iterates thru all the <sent_tag>s
        for sent in root.findall('sent/sent_tag'):

            # write the binary label for this sentence
            binary_label = 0 if random.random() < zero_proportion else 1
            output_fp.write(str(binary_label))

            # write the tab-separator
            output_fp.write('\t')

            # write the characters for this sentence
            for line in sent.text.splitlines():
                line = line.strip() # remove whitespace
                if not line:        # empty lines
                    continue
                for char in line:
                    # stop writing when we encounter the '/' character
                    if char == '/':
                        break
                    # FIXME: should this check hanzidentifer.has_chinese(char) ??
                    output_fp.write(char)

            # write the newline-separator
            output_fp.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files',
            type=argparse.FileType('r'),    # readable file
            required=True,
            nargs='+',                      # one or more
            help='raw hkcancor files to preprocess')
    parser.add_argument('-o', '--output-file',
            type=argparse.FileType('w'),    # writable file
            required=True,
            help='file to write tab-separated sentences to')
    parser.add_argument('--proportion',
            type=float,
            default=0.3,
            help='proportion of sentences to mark as "0" (as opposed to "1")')
    args = parser.parse_args()

    if args.proportion < 0 or 1 < args.proportion:
        raise ValueError('proportion must be in range [0.0, 1.0]')
    preprocess(args.input_files, args.output_file, args.proportion)

