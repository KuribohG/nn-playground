import os
import gzip
import pickle

def get_data(environ=False):
    path = os.environ["MNIST_PKL_GZ"] if environ else "../mnist_data"
    if not os.path.exists(path):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, path)

    f = gzip.open(path, 'rb')
    try:
        split = pickle.load(f, encoding="latin1")
    except TypeError:
        split = pickle.load(f)
    f.close()

    which_sets = "train valid test".split()
    return dict((which_set, dict(features=x.astype("float32"),
                                 targets=y.astype("int32")))
                for which_set, (x, y) in zip(which_sets, split))
