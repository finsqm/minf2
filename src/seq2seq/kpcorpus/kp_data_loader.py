import os
import csv
import copy
import logging
import sys
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

DATA_DIR_NAME = 'MINF_DATA_DIR'
CORPUS_NAME = 'kpcorpus/csv'

TYPE_INDEX = 2
PITCH_INDEX = 4
TIME_INDEX = 1
VELOCITY_INDEX = 5
CHORD_INDEX = 3
KEY_INDEX = 3
MODE_INDEX = 4


def transform_to_X_Y(X, y):
    """
    Input:
        X : 2D
        y : 1D
    Output:
        X : 1D
        Y : 1D
    """
    X_out = []
    Y_out = []
    for i, x in enumerate(X):
        for x_j in x:
            X_out.append(x_j)
            Y_out.append(y[i])
    return X_out, Y_out


def to_one_hot_vector(x, n):
    """
    Input :
        x : scalar
        n : dim
    Output :
        vetor: 1D one hot numpy array of size n
    """
    vector = np.zeros((n,), dtype=np.int)
    vector[int(x)] = 1
    return vector


def sequence(XX, YY):
    """
    Inputs:
        XX          : 2D
        XX[:]       : songs
        XX[:][:]    : notes

        YY          : 2D
        YY[:]       : songs
        YY[:][:]    : chords

    Output:
        X          : 2D
        X[:]       : notes
        X[:][:]    : note vector

        Y[:]       : chords

        L[:]       : song lengths
    """
    X = []
    Y = []
    L = []
    note_dim = 12
    for i, XX_song in enumerate(XX):
        counter = 0
        for j, XX_note in enumerate(XX_song):
            counter += 1
            note_vec = to_one_hot_vector(XX_note, note_dim)
            X.append(note_vec)
            Y.append(YY[i][j])
        L.append(counter)

    return np.asarray(X), np.asarray(Y), np.asarray(L)


class Note(object):
    def __init__(self, note_on, pitch):
        self.note_on = note_on
        self.pitch = pitch
        self.pc = pitch % 12
        self.on = True

    def set_note_off(self, note_off):
        if self.on:
            self.duration = note_off - self.note_on
            if self.duration < 0:
                raise Exception('Note Off before Note On')
            self.on = False

    def is_on(self):
        return self.on


class Chord(object):
    def __init__(self, time, chord_string):
        self.time = time
        self.chord_string = chord_string
        self.pc = None
        # Set self.pc
        self.set_pc(chord_string)

    def set_pc(self, chord_string):

        key = chord_string[0]
        modifier = chord_string[1]
        minor = 'min' in chord_string

        if modifier == '#':
            m = 1
        elif modifier == 'b':
            m = -1
        else:
            m = 0

        k = key.capitalize()
        d = ord(k) - 67

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
        # if minor:
        #     pc = (pc + 3) % 12

        self.pc = pc


class Song(object):
    def __init__(self):
        self.notes = dict()
        self.notes_list = []
        self.chord_list = []

    def transpose_to_c(self, note):
        if self.mode == 'minor':
            return (note - self.key - 3) % 12
        else:
            return (note - self.key) % 12

    def transpose_chord_to_c(self, note):
        if self.mode == 'minor':
            return (note - self.key - 3) % 12
        else:
            return (note - self.key) % 12

    def get_X_and_y(self):
        """
        X :	2D
            X[:] 		= frames (varying size)
            X[:][:]		= notes
        y :	1D
            y[:]	= Labels

        Note: assumes ordered chord and note list
        """
        X = []
        y = []
        notes_list_copy = copy.copy(self.notes_list)
        chord_list_copy = self.chord_list[1:]

        y.append(to_one_hot_vector(self.transpose_chord_to_c(self.chord_list[0].pc), 12))
        for chord in self.chord_list[1:]:
            x = []
            # print chord.pc, self.transpose_chord_to_c(chord.pc), self.key
            y.append(to_one_hot_vector(self.transpose_chord_to_c(chord.pc), 12))
            time = chord.time
            while notes_list_copy[0].note_on < time:
                note = notes_list_copy.pop(0)
                x_note = self.transpose_to_c(note.pc)
                x.append(to_one_hot_vector(x_note, 12))
            X.append(x)

        x = []
        while len(notes_list_copy) > 0:
            note = notes_list_copy.pop(0)
            x_note = self.transpose_to_c(note.pc)
            x.append(to_one_hot_vector(x_note, 12))
        X.append(x)
        return X, y


class KPDataLoader(object):
    def __init__(self):
        self.data_folder = os.path.join(os.environ[DATA_DIR_NAME], CORPUS_NAME)
        self.songs = []

    def load_file(self, filename):
        data_path = os.path.join(self.data_folder, filename)
        song = Song()

        with open(data_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[TYPE_INDEX].strip() == 'Note_on_c':
                    pitch = float(row[PITCH_INDEX].strip())
                    time = float(row[TIME_INDEX].strip())
                    try:
                        if song.notes[pitch][-1].is_on():
                            song.notes[pitch][-1].set_note_off(time)
                        else:
                            note = Note(time, pitch)
                            song.notes[pitch].append(note)
                            song.notes_list.append(note)
                    except KeyError:
                        song.notes[pitch] = []
                        note = Note(time, pitch)
                        song.notes[pitch].append(note)
                        song.notes_list.append(note)

                if row[TYPE_INDEX].strip() == 'Lyric_t':
                    time = float(row[TIME_INDEX].strip())
                    chord_string = row[CHORD_INDEX].strip().strip('"')
                    chord = Chord(time, chord_string)
                    song.chord_list.append(chord)

                if row[TYPE_INDEX].strip() == 'Key_signature':
                    song.key = int(row[KEY_INDEX])
                    song.mode = row[MODE_INDEX]

        self.songs.append(song)

    def get_XX_and_YY(self):
        if len(self.songs) > 0:
            XX = []
            YY = []
            max_length = 0
            for song in self.songs:
                x, y = song.get_X_and_y()
                X, Y = transform_to_X_Y(x, y)
                length = len(X)
                if length > max_length:
                    max_length = length
                XX.append(X)
                YY.append(Y)
            return XX, YY, max_length
