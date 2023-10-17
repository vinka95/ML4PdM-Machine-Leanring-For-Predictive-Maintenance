import pytest

from ml4pdm.data import ANY, DATETIME, MULTIDIMENSIONAL, NUMERIC, SETOFLABELS, TIMESERIES, AttributeType


def test_attribute_parsing():
    assert isinstance(AttributeType.parse('NUMERIC'), NUMERIC)
    assert str(NUMERIC()) == 'NUMERIC'
    assert isinstance(AttributeType.parse('DATETIME'), DATETIME)
    assert str(DATETIME()) == 'DATETIME'
    sol = AttributeType.parse('{A,B,C}')
    assert isinstance(sol, SETOFLABELS)
    assert sol.setoflabels == {'A', 'B', 'C'}
    md = AttributeType.parse('MULTIDIMENSIONAL[2,2,3]')
    print(md)
    assert isinstance(md, MULTIDIMENSIONAL)
    assert str(md) == 'MULTIDIMENSIONAL[2, 2, 3]'
    assert md.dimensions == [2, 2, 3]
    ts = AttributeType.parse('TIMESERIES(NUMERIC:NUMERIC)')
    assert isinstance(ts, TIMESERIES)
    assert isinstance(ts.timetype, NUMERIC)
    assert isinstance(ts.valuetype, NUMERIC)
    assert repr(ANY()) == "ANY"


def test_attribute_errors():
    with pytest.raises(TypeError):
        AttributeType.parse('error')

    with pytest.raises(TypeError):
        TIMESERIES.parse('TIMESERIES({A,B}:NUMERIC)')

    with pytest.raises(TypeError):
        TIMESERIES.parse('TIMESERIES(NUMERIC:error)')

    with pytest.raises(ValueError):
        MULTIDIMENSIONAL.parse('MULTIDIMENSIONAL[-1,1]')

    with pytest.raises(TypeError):
        TIMESERIES(DATETIME(), DATETIME)

    with pytest.raises(TypeError):
        SETOFLABELS(4)

    with pytest.raises(TypeError):
        MULTIDIMENSIONAL(9)
