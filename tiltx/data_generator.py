import numpy


class DataGenerator(object):
    """Generate data to work with the package."""

    def __init__(self,):
        pass

    @staticmethod
    def example(i):
        """Recreate the values from Example i.
        The data is stored in data/data_i.txt for i in {1,...,6}.
        :returns: tuple(t, angle, angle)
        :rtypes: (array, array, array)
        """
        with numpy.errstate(divide='ignore'):
            d = numpy.loadtxt("data/data_"+str(i)+".txt")
            return d[:,0], d[:,1], d[:,2]
        
 