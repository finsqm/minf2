import os
import csv
import numpy as np

class DataLoader(object):

    def __init__(self):
        self.load()

    def load(self, filename='by_note.csv'):
        '''
        Load Weimar Jazz DB Data from pre-processed csv file
        '''
        data_path = os.path.join(os.environ['MINF_DATA_DIR'], filename)

        with open(data_path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                
            self.data = reader
