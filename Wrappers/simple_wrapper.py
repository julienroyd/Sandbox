import numpy as np


class ObjectWrapper(object):
    def __init__(self, obj):
        self.obj = obj
    
    def __getattr__(self, name, *args, **kwargs):
        attr = getattr(self.obj, name)
        assert hasattr(attr, '__call__'), f'{name} is not callable'
        def newfunc(*args, **kwargs):
            print(f'calling {attr.__name__}, with args={args}, kwargs={kwargs}')
            result = attr(*args, **kwargs)
            print(f'done calling {attr.__name__}')
            return result
        return newfunc


if __name__ == '__main__':
    x = np.array([1, 2, 3])
    x = ObjectWrapper(x)
    print(x.sum(axis=0))
