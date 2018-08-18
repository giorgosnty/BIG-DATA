#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
	
	# remove trailing spaces at end of each line
	word = line.strip()

	# print mapper output to STOUT (standard output): key <tab> value
    print "%s\t%s" % (word,1)
