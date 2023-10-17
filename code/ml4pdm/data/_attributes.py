class AttributeType():
    """Abstract AttributeType class, which is used as a parent class of all attribute/feature type classes.
    These are NUMERIC, TIMESERIES, DATETIME, MULTIDIMENSIONAL, SETOFLABELS
    """
    @staticmethod
    def parse(type_string: str):
        """Parses the given type_string to an attribute type.

        :param type_string: The string to be parsed to an attribute type
        :type type_string: str
        :raises TypeError: If the specified type_string is not recognized
        :return: Attribute type that was parsed from the given type_string
        :rtype: Subclass of AttributeType
        """
        if type_string[:1] == "{" and type_string[-1:] == "}":
            return SETOFLABELS.parse(type_string)
        if type_string.startswith('TIMESERIES'):
            return TIMESERIES.parse(type_string)
        if type_string == 'NUMERIC':
            return NUMERIC()
        if type_string == 'DATETIME':
            return DATETIME()
        if type_string.startswith('MULTIDIMENSIONAL'):
            return MULTIDIMENSIONAL.parse(type_string)
        raise TypeError('The type_string is not recognized as an attribute type!')


class ANY(AttributeType):
    def __repr__(self):
        return 'ANY'


class NUMERIC(AttributeType):
    def __repr__(self):
        return 'NUMERIC'


class SETOFLABELS(AttributeType):
    def __init__(self, setoflabels):
        if not isinstance(setoflabels, set):
            raise TypeError('Creating SETOFLABELS requires the setoflabels parameter to be a set!')
        self.setoflabels = setoflabels

    def __repr__(self):
        return str(self.setoflabels)

    @classmethod
    def parse(cls, type_string):
        return cls(set(type_string.strip('{} ').split(',')))


class DATETIME(AttributeType):
    def __repr__(self):
        return 'DATETIME'


class TIMESERIES(AttributeType):
    def __init__(self, timetype, valuetype):
        if not isinstance(timetype, NUMERIC) and not isinstance(timetype, DATETIME) or not issubclass(type(valuetype), AttributeType):
            raise TypeError('Creating TIMESERIES type requires the timetype to be NUMERIC or DATETIME and the valuetype to be an AttributeType')
        self.timetype = timetype
        self.valuetype = valuetype

    def __repr__(self):
        return 'TIMESERIES('+str(self.timetype)+':'+str(self.valuetype)+')'

    @classmethod
    def parse(cls, type_string):
        type_string = type_string.split('(', 1)
        step, value = type_string[1].split(':')
        value = value.rstrip(')')
        return cls(super().parse(step), super().parse(value))


class MULTIDIMENSIONAL(AttributeType):
    def __init__(self, dimensions):
        if not hasattr(dimensions, '__len__'):
            raise TypeError('Creating MULTIDIMENSIONAL type requires the dimensions to have a length!')
        self.dimensions = dimensions
        if any(dim <= 0 for dim in self.dimensions):
            raise ValueError('Creating MULTIDIMENSIONAL type requires every dimensionvalue to be positive!')

    def __repr__(self):
        return 'MULTIDIMENSIONAL'+str(self.dimensions)

    @classmethod
    def parse(cls, type_string):
        return cls(list(map(int, type_string[16:].strip(' []').split(','))))
