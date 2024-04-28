import functools

def mydecorator( f ):
   @functools.wraps( f )
   def wrapper( *args, **kwargs ):
      print(f"Calling f. args={args}, kwargs={kwargs}")
      v = f( *args, **kwargs )
      print(f"f returned v={v}")
      return v
   return wrapper


@mydecorator
def add(x, y):
	return x + y

print(add(5, 7))
