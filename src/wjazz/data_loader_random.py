import os
import csv
import numpy as np

DATA_DIR_NAME = 'MINF_DATA_DIR'
CORPUS_NAME = 'wjazz'
TPC_DIM = 12

def to_one_hot_vector(x, n):
	"""
	Input :
		x : scalar
		n : dim
	Output :
		vetor: 1D one hot numpy array of size n
	"""
	vector = np.zeros((n,), dtype=np.int)
	vector[x] = 1
	return vector

def process_chord_string(chord_raw, key=None):
    tonic = chord_raw[0]
    try:
        accidental = chord_raw[1]
    except IndexError:
        accidental = None

    k = tonic.capitalize()
    d = ord(k) - 67

    if accidental == '#':
        m = 1
    elif accidental == 'b':
        m = -1
    else:
        m = 0

    # For A and B
    if d < 0:
        d = 7 + d

    # C,D,E
    if d < 3:
        pc = 2 * d
    # F,G,A,B
    else:
        pc = (2 * d) - 1

    pc = (pc + m) % 12

    if key:
        key_pc = process_chord_string(key)
        tpc = (pc - key_pc) % 12
    else:
        tpc = pc

    return tpc + 1

class DataLoader(object):

    def __init__(self):
        self.load()

    def load(self, filename='by_note_cleaned.csv'):
        '''
        Load Weimar Jazz DB Data from pre-processed csv file
        '''
        data_folder = os.path.join(os.environ['MINF_DATA_DIR'], CORPUS_NAME)
        data_path = os.path.join(data_folder, filename)

        with open(data_path) as f:
            reader = csv.DictReader(f, delimiter=';')
            counter = 0
            song_name = None
            X = []
            Y = []
            L = []
            for row in reader:
                if row['filename_sv'] != song_name:
                    if song_name:
                        L.append(counter)
                    counter = 0
                    song_name = row['filename_sv']
                counter += 1

                chord_raw = row['chords_raw']
                tpc_raw = row['tpc_raw']
                key_raw = row['key']

                chord = process_chord_string(chord_raw, key_raw)
                tpc = int(tpc_raw)
                tpc_vec = to_one_hot_vector(tpc, TPC_DIM)

                X.append(tpc_vec)
                Y.append(chord)

            L.append(counter)

            X = np.asarray(X)
            Y = np.asarray(Y)
            L = np.asarray(L)

            return X, Y, L
