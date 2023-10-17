from ml4pdm.transformation._transformer import Transformer


class FixedSizeFeatureExtractor(Transformer):
    """
    This is an abstract subclass for all transformer classes that transform a univariate timeseries feature into a fixed size feature vector.
    Classes that inherit from this class need to implement a fit method and a transform method.
    """
