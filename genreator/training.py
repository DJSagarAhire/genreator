def create_training_set(movies_with_images_dataset):
	""" Accepts a dataset with flattened images and genres for movies and returns a scikit-compatible training set and additional info.

	X contains an ordered list of image arrays.
	y contains an ordered list of vectorized genre arrays.
	Also returns the multilabel binarizer to decode the results.
	"""

	from sklearn.preprocessing import MultiLabelBinarizer

	X = [movie['flattened_poster'] for movie in movies_with_images_dataset]
	genres = [movie['genres'] for movie in movies_with_images_dataset]
	mlb = MultiLabelBinarizer()
	y = mlb.fit_transform(genres)

	return X, y, mlb

def train_classifier(X, y, classifier=None):
	""" Creates and trains a classifier on the given dataset and returns it.

	If classifier is already supplied, uses it, else creates a random forest classifier.
	"""

	if classifier is None:
		from sklearn.ensemble import RandomForestClassifier
		classifier = RandomForestClassifier(n_estimators=50, oob_score=True)

	classifier.fit(X, y)

	return classifier

def test_accuracy(X, y, classifier, k=5):
	""" Tests the accuracy of the classifier using k-fold cross validation.
	"""

	from sklearn.model_selection import cross_val_score
	from sklearn.metrics import hamming_loss, make_scorer

	hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)
	scores = cross_val_score(classifier, X, y, scoring=hamming_scorer, cv=k)
	return 1 + scores.mean()