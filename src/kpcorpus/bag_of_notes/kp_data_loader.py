import os
import csv
import copy
import logging
import sys

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

class Note(object):
    def __init__(self, note_on, pitch):
        self.note_on = note_on
        self.pitch = pitch
        self.pc = int(pitch % 12)
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
        if minor:
            pc = (pc + 3) % 12

        self.pc = int(pc + 1)

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
            return ((note - 1 - self.key - 3) % 12) + 1
        else:
            return ((note - 1 - self.key) % 12) + 1

    def get_X_and_y(self):
        '''
        X :	3D
            X[:] 		= frames (varying size)
            X[:][:]		= notes
            X[:][:][:]	= components
        y :	1D
            y[:]	= Labels

        Note: assumes ordered chord and note list
        '''
        X = []
        y = []
        notes_list_copy = copy.copy(self.notes_list)
        chord_list_copy = self.chord_list[1:]

        y.append(self.chord_list[0].pc)
        for chord in self.chord_list[1:]:
            x = []
            y.append(self.transpose_chord_to_c(chord.pc))
            time = chord.time
            while notes_list_copy[0].note_on < time:
                note = notes_list_copy.pop(0)
                x_note = [self.transpose_to_c(note.pc), note.duration]
                x.append(x_note)
            X.append(x)

        x = []
        while len(notes_list_copy) > 0:
            note = notes_list_copy.pop(0)
            x_note = [self.transpose_to_c(note.pc), note.duration]
            x.append(x_note)
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

    def get_XX_and_Y(self):
        if len(self.songs) > 0:
            XX = []
            Y = []
            for song in self.songs:
                X, y = song.get_X_and_y()
                XX.append(X)
                Y.append(y)
        return XX, Y
