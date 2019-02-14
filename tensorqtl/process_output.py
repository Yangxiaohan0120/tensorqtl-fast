#!/bin/env/python
#-*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function
from __future__ import division
import numpy as np
import sys
import os

def main():
	with open('output.txt','r') as f:
		data = f.read()
		data = data.split('\n')[5:]

		ncalls   = []
		tottime  = []
		percall  = []
		cmtime   = []
		_percall = []
		filename = []

		_types = [int,float,float,float,float,str]

		for n in data:
			n = n.split(' ')
			n = [z for z in n if z != '']
			if len(n) > 5:
				n[5] = ' '.join(n[5:])
				n = n[:6]
				for i,z in enumerate([ncalls,tottime,percall,
						cmtime,_percall,filename]):
					if not i:
						if '/' in n[i]:
							n[i] = n[i].split('/')[0]
					z.append(_types[i](n[i]))

		tottime,filename = zip(*sorted(zip(tottime,filename)))

		for name,_time in zip(filename,tottime):
			print('{}\t\t{}'.format(_time,name))


if __name__ == "__main__":
	main()

