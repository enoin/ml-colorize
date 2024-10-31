class ImageProperties(object):
    def __init__(self, gray, ground_truth, colorized=None):
        self._gray = gray
        self._colorized = colorized
        self._ground_truth = ground_truth

    @property
    def colorized(self):
        return self._colorized

    @property
    def gray(self):
        return self._gray

    @property
    def ground_truth(self):
        return self._ground_truth

    @colorized.setter
    def colorized(self, colorized):
        self._colorized = colorized

