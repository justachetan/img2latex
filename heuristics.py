class segment(object):

	def __init__(self, ch, x, y, w, h):
		# Character
		self.ch = ch
		# x coordinate of centroid of bounding box
		self.x = x
		# y coordinate of centroid of bounding box
		self.y = y
		# Width of bounding box
		self.w = w
		# Weight of bounding box
		self.h = h



# Function that returns the LaTeX code corresponding to the segments.
def getLatex(segments):

	expr = []
	# Assuming that the first segment corresponds to a character that is... 'normal' script.
	expr.append(segments[0].ch)

	for i in range(1, len(segments)):
		curr = segments[i]
		prev = segments[i - 1]

		# = test
		if curr.ch == '-' and prev.ch == '-':
			if curr.x - curr.w / 2 < prev.x + prev.w / 2:
				expr[len(expr) - 1] = '='

		# Subscript test
		elif curr.y < prev.y - prev.h / 2:
			expr.append('_{')
			expr.append(curr.ch)
			expr.append('}')

		# Superscript test
		elif curr.y > prev.y + prev.h / 2:
			expr.append('^{')
			expr.append(curr.ch)
			expr.append('}')

		else:
			expr.append(curr.ch)

	return ''.join(expr)
