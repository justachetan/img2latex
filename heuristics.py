class segment(object):

	def __init__(self, ch, x, y, w, h):
		# Character
		self.ch = ch
		# x coordinate of centroid of bounding box; increases from left to right
		self.x = x
		# y coordinate of centroid of bounding box; increases from top to bottom
		self.y = y
		# Width of bounding box
		self.w = w
		# Weight of bounding box
		self.h = h



def latex_sqrt(seg, i, expr):
	expr.append('\\sqrt')
	root = seg[i]
	j = i + 1
	while j < len(seg):
		if seg[j].x < root.x + root.w / 2:
			expr.append(seg[j].ch)
			i += 1
	expr.append('}')
	return i


def latex_horizontal(seg, i, expr):

	# Equal to
	if i + 1 < len(seg) and seg[i + 1].ch == '-':
		if seg[i + 1].x - seg[i + 1].w / 2 < seg[i].x + seg[i].w / 2:
			expr.append('=')
			return i + 2

	# # Division
	# if i + 2 < len(seg) and seg[i + 1].ch == '.' and seg[i + 2].ch == '.':
	# 	if seg[i + 1].x < seg[i].x + seg[i].w / 2 and seg[i + 2].x < seg[i].x + seg[i].w / 2:
	# 		if (seg[i + 1].y > seg[i].y and seg[i + 2].y < seg[i].y) or (seg[i + 1].y < seg[i].y and seg[i + 2].y > seg[i].y)
	# 			expr.append(' \div ')
	# 			return i + 3

	# Fraction
	if i + 2 < len(seg):
		bar = seg[i]
		num = []
		den = []
		j = i + 1
		while j < len(seg) and seg[j].x < bar.x:
			if(seg[j].y < bar[j].y):
				num.append(seg[j].ch)
			else:
				den.append(seg[j].ch)
		if num != [] and den != []:
			expr.append('\\frac{' + ''.join(num) + '}{' + ''.join(den) + '}')
			return j

	# Minus
	expr.append('-')
	return i + 1


def latex_subscript(seg, i, expr):
	if seg[i + 1].y > seg[i].y + seg[i].h / 2:
		base = seg[i]
		j = i + 1
		expr.append('_{')
		while j < len(seg) and seg[j].y > base.y + base.h / 2:
			expr.append(seg[j].ch)
		expr.append('}')
		return j
	return i


def latex_superscript(seg, i, expr):
	if seg[i + 1].y + seg[i + 1].h / 2 < seg[i].y:
		base = seg[i]
		j = i + 1
		expr.append('^{')
		while j < len(seg) and seg[j].y + seg[j].h / 2 < base.y:
			expr.append(seg[j].ch)
		expr.append('}')
		return j
	return i



# Function that returns the LaTeX code corresponding to the segments.
def get_latex_expr(segments):

	expr = []
	i = 0

	while i < len(segments):

		curr = segments[i]
		prev = segments[i - 1]

		# -
		if curr.ch == '-':
			i = latex_horizontal(segments, i, expr)

		# sqrt
		elif curr.ch == 'sqrt':
			i = latex_sqrt(segments, i, expr)


		else:
			expr.append(curr.ch)
			i_prev = i
			i = expr.latex_subscript(segments, i, expr)
			if(i == i_prev):
				i = expr.latex_superscript(segments, i, expr)
			if(i == i_prev):
				i += 1

	return ''.join(expr)
