def predict_genre(image, classifier, mlb):
	""" Accepts an image as a numpy array and returns a list of its movie genre.

	Accepts the classifier and MultiLabelBinarizer used as input.
	"""

	from preprocessing import normalize_single_image

	flattened_image = normalize_single_image(image)[1]
	genres_indicator = classifier.predict(flattened_image.reshape(1,-1))
	return mlb.inverse_transform(genres_indicator)

def load_image_from_path(image_path):
	""" Accepts a path to an image and returns the numpy array corresponding to it.
	"""

	import imageio
	return imageio.imread(image_path)