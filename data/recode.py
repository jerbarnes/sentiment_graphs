#!/bin/env python3
# coding: utf-8

# Replace broken Unicode characters with real ones

import sys
import re

def replace_chars(match):
    char = match.group(0)
    return chars[char]

chars = {
    '\x82' : ',',        # High code comma
    '\x84' : ',,',       # High code double comma
    '\x85' : '...',      # Tripple dot
    '\x88' : '^',        # High carat
    '\x91' : '\x27',     # Forward single quote
    '\x92' : '\x27',     # Reverse single quote
    '\x93' : '\x22',     # Forward double quote
    '\x94' : '\x22',     # Reverse double quote
    '\x95' : ' ',
    '\x96' : '-',        # High hyphen
    '\x97' : '--',       # Double hyphen
    '\x99' : ' ',
    '\xa0' : ' ',
    '\xa6' : '|',        # Split vertical bar
    '\xab' : '<<',       # Double less than
    '\xbb' : '>>',       # Double greater than
    '\xbc' : '1/4',      # one quarter
    '\xbd' : '1/2',      # one half
    '\xbe' : '3/4',      # three quarters
    '\xbf' : '\x27',     # c-single quote
    '\xa8' : '',         # modifier - under curve
    '\xb1' : ''          # modifier - under line
}


text = open(sys.argv[1], "r").read()

out = re.sub('(' + '|'.join(chars.keys()) + ')', replace_chars, text)

print(out)

