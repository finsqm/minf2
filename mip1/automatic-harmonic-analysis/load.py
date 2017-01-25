def load(self,csv_file_name):

		raw_XX = [] # 3D list (2nd dim is mutable)
		raw_Y = []  # 2D list (2nd dim is mutable)

		with open(csv_file_name) as csv_file:
			reader = csv.DictReader(csv_file,delimiter=';')
			past_name = None
			X = []
			y = []

			for row in reader:
				# Each row corresponds to a frame (bar)
				# Using 'filename_sv' to determine song boundaries
				if past_name != row['filename_sv']:
					if X:
						raw_XX.append(X)
					if y:
						raw_Y.append(y)

					X = []
					y = []

				past_name = row['filename_sv']

				# Get rid of songs with no key
				if not row['key']:
					continue

				# Note: mode not currently used
				key, mode = self._process_key(row['key'])
				self.keys.append(key)
				X_i = self._process_Xi(row['tpc_raw'],row['metrical_weight'])
				y_i = self._process_yi(row['chords_raw'],row['chord_types_raw'],key)


				# get rid of bars with no chords
				if not y_i:
					continue

				X.append(X_i)
				y.append(y_i)

			if X:
				raw_XX.append(X)
			if y:
				raw_Y.append(y)

		self.XX = self._process_XX(raw_XX) 	# 4D
		self.Y = self._process_Y(raw_Y) 	# 2D