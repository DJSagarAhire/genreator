def load_dataset(dataset_file_path):
    """ Accepts a file path to a dataset and returns the dataset in the form of a list of dicts.

    The dataset is assumed to be a csv with the columns: 'genres', 'movie_title' and 'imdb_title'.
    The genre is assumed to be a string with each genre separated by pipes (example: 'Action|Adventure|Fantasy|Sci-Fi')
    """

    import csv

    movies_dataset = []

    with open(dataset_file_path) as dataset_file:
        reader = csv.DictReader(dataset_file)
        for movie in reader:
            movie['genres'] = movie['genres'].split('|')
            movies_dataset.append(movie)

    # TODO: Throw error if the dicts don't match the specs

    return movies_dataset

def load_all_images(movies_dataset):
    """ Accepts a movie dataset and returns images corresponding to each as additional entries in a copy of the dataset dict.

    Images are sent as numpy arrays.
    The original dataset dict is left unchanged. A copy is returned instead.
    Calls load_single_image to handle each individual image.
    """

    movies_with_images_dataset = []
    for movie in movies_dataset:
        new_movie = movie.copy()
        curr_image = load_single_image(movie['imdb_title'])
        if curr_image is not None:
            movies_with_images_dataset.append(new_movie)
        new_movie['poster'] = curr_image

    return movies_with_images_dataset

def load_single_image(imdb_title, attempt_download=False):
    """ Accepts an IMDB title string and returns the image corresponding to it as a numpy array.

    The image is assumed to be in the directory 'data/images/<IMDB Title>.jpg'.
    If it doesn't exist, download_image is called to download it off the internet.
    """

    import imageio
    import os.path

    image_path = 'data/images/{}.jpg'.format(imdb_title)

    if not os.path.isfile(image_path):
        if not attempt_download:
            return None

        status_code = download_image(imdb_title)
        if status_code != 0:
            return None

    img_array = imageio.imread(image_path)
    return img_array

def normalize_images(movies_with_images_dataset):
    """ Accepts a dataset with movies and images and performs normalizations on each of the images in the list.

    Each dict in the list is assumed to have a 'poster' field with the raw image array.
    The same list is returned with 'normalized_poster' and 'flattened_poster' fields added to each dict.
    """

    for movie in movies_with_images_dataset:
        curr_image = movie['poster']
        movie['normalized_poster'], movie['flattened_poster'] = normalize_single_image(curr_image)

    return movies_with_images_dataset

def normalize_single_image(image, target_size=(148, 100)):
    """ Accepts an image as a numpy array and performs normalizations on it.

    The current normalization performed is resizing to the specified size and flattening the image.
    Returns both the resized and flattened images. 
    """

    from skimage.transform import resize

    resized_image = resize(image, target_size)
    flattened_image = resized_image.flatten()

    return resized_image, flattened_image

def download_image(imdb_title):
    """ Accepts an IMDB title and downloads the image to 'data/images/<IMDB Title>.jpg'. Returns an integer indicating success / failure.

    Uses the OMDB API to get the image URL and requests to subsequently dowload the image.
    If the directory 'data/images does not exist, it is created automatically.'
    If the download fails, a non-zero integer is returned as an error code.

    Error codes:
    1: The OMDB API supplied 'Poster' as 'N/A'
    2: The response status code is non-200
    3: The image download is unsuccessful (imread throws an HTTPError)
    """

    import requests
    import imageio
    import os
    import time
    from urllib.error import HTTPError

    response = requests.get('http://www.omdbapi.com/', params={'i': imdb_title})

    if response.status_code != 200:
        print('Error: response from server: {}'.format(response.status_code))
        return 2

    poster_url = response.json()['Poster']
    if poster_url == 'N/A':
        print('Error: No Poster URL returned from server for title {0}'.format(imdb_title))
        return 1

    try:
        img_array = imageio.imread(poster_url)
    except HTTPError as http_err:
        print('Error: {0}'.format(http_err))
        return 3

    os.makedirs('data/images', exist_ok=True)
    imageio.imwrite('data/images/{0}.jpg'.format(imdb_title), img_array)
    time.sleep(0.5)         # Sleeps to prevent too many API requests

    return 0
