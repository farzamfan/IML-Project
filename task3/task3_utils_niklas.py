import itertools
import numpy as np

def sequences_to_one_hot(sequences):
#expects array with sequences as elements

    sequences_one_hot = np.zeros(len(sequences),dtype='object')

    amino_alphabet = ['R','H','K','D','E','S','T','N','Q','C','U','G','P','A','I','L','M','F','W','Y','V']

    for i in np.arange(len(sequences)):

        sequences_one_hot[i] = np.asarray([0 if elem != elem_seq else 1 for elem_seq in sequences[i] for elem in amino_alphabet])

    return sequences_one_hot
