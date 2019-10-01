import numpy
import pytest
from tiltx.data_generator import DataGenerator
from tiltx.feature_extractor import FeatureExtractor


@pytest.mark.parametrize("RT_method", ['cumsum'])
def test_example1(RT_method):
    """ """
    t, alpha, beta = DataGenerator.example(1)
    features = FeatureExtractor(t,alpha,beta,RT_method='cumsum',correct='up')
    assert numpy.isclose(features.RT, 0.66, rel_tol=0.05)

