# Databases

## Table of Contents
- [Singletons](#singletons)
- [Real-world Example](#real-world-example)
- [Decorator](#decorator)
- [Database Connectors](#database-connectors)
- [Context Managers & Decorators](#context-managers--decorators)
- [Handling Credentials](#handling-credentials)

## Singletons

The singleton design pattern is a simple pattern that allows you to share a single resource or best put, manage a shared resource. It allows you to instantiate just a single instance of your class and restrict multiple instantiations of the said class.

You would use a singleton when you want to:
- Retrieve and store information on external configuration files
- Manage the connection to a database or external files or drivers, etc.

### Real-world Example

Imagine that an application on a server has several parts creating its own connection to a database and also performing different operations on the database, chaotic right?

A singleton allows you to create just one instance of the connection to the database and that same object is used to perform other operations on the database in all the parts of the application.

**In summary, Singleton ensures that a class has only one instance and provides a global point of access to it.**

There are several singletons in Python that you use frequently, including `None`, `True`, and `False`. It is the fact that `None` is a singleton that allows you to compare for `None` using the `is` keyword.

In Python, the typical way of creating a singleton pattern is using a decorator.

Suppose we want a single database connection throughout the life-cycle of our application. Let's create a singleton class to serve our purpose.

```python
class Singleton:
    def __init__(self, cls):
        self._cls = cls
    
    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance
    
    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')
    
    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)

@Singleton
class DBConnection(object):
    def __init__(self):
        """Initialize your database connection here."""
        pass
    
    def __str__(self):
        return 'Database connection object'

c1 = DBConnection.Instance()
c2 = DBConnection.Instance()
print("Id of c1 : {}".format(str(id(c1))))
print("Id of c2 : {}".format(str(id(c1))))
print("c1 is c2 ? " + str(c1 is c2))

# output
# Id of c1 : 139699882512960
# Id of c2 : 139699882512960
# c1 is c2 ? True
```

## Decorator

Reference: [Database Connections in Python: Extensible, Reusable, and Secure by Eric Pan](https://www.datacamp.com/tutorial/database-connections-python)

## Database Connectors

Imagine you're a data scientist in charge of building a model, and you need to connect to a database — a pretty common task for you. How would you do it?

The easiest way is probably something like this:

```python
import cx_Oracle
import pandas as pd

cnn = cx_Oracle.connect(conn_str)
data = pd.read_sql(sql, cnn.connector)
# or do something else
```

(We'll use an Oracle connection throughout the post as an example, but this blog applies to all other databases as well!)

This little code snippet seems like an obvious choice, right? This approach might get the job done, but isn't the best way to go about it.

### Why?

Well, hopefully we were instructed to close any pointers or connectors after using them. Database connections are a limited and relatively expensive resource. If you're not careful, you may accidentally hit the maximum number of connections with only one running project, but dozens of connections from historical projects.

A more careful way to deal with database connectors might be:

```python
import cx_Oracle
import pandas as pd

cnn = cx_Oracle.connect(conn_str)
data = pd.read_sql(sql, cnn.connector)
# or do something else
cnn.close()
```

Seems solid, right? Well, it's better, but not quite there.

Not sure why? Consider the following: what happens if something in the middle interrupts the script? With this approach, you could accidentally reach the maximum number of allowable connections if, say, several testing scripts have database operations fail during execution. (Some other relevant questions here: what if you changed the name of the engine/connector and forgot to change it subsequent lines of code? Or if you have multiple connectors at the same time, but forget about one of them?)

If you're really cautious and want to solve for each of these cases, you may come up with something like this:

```python
import cx_Oracle
import logging

try:
    cnn = cx_Oracle.connect(conn_str)
    cnn.do_something()
except Exception:
    cnn.rollback()
    logging.error("Database connection error")
    raise
else:
    cnn.commit()
finally:
    cnn.close()
```

Now this looks good. We employ a rollback if anything goes wrong, commit if things go right, and close our connection in the end. Though this is certainly a significant improvement over our first approach, we're still short of best practices.

Why? Well, sometimes you have to deal with multiple database connectors (in the data exploration phase, for example), and you'll have to read from each of them on a regular basis. This might introduce some redundancy, which suggests that a reusable function might be a good fit for this scenario.

## Context Managers & Decorators

The best practices discussed above remind me of Python file objects: simply initialize a connection and safely close it when you're done, regardless of what happens in the middle. With this syntax, the file is properly closed after the block finishes, even if an exception is raised on the way. It'd be great if we could handle database connections in a comparable manner.

Though we can't treat connecting to a database in exactly the same way, it turns out that we can use something similar known as a context manager:

```python
import cx_Oracle

class oracle_connection(object):
    """oracle db connection"""
    def __init__(self, connection_string=conn_str):
        self.connection_string = connection_string
        self.connector = None
    
    def __enter__(self):
        self.connector = cx_Oracle.connect(self.connection_string)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.connector.commit()
        else:
            self.connector.rollback()
        self.connector.close()
```

Context managers are used for managing resources. They allow you to allocate and release resources precisely when you wish. With this particular context manager, we can call database connections like so:

```python
import pandas as pd

with oracle_connection() as conn:
    data = pd.read_sql(sql, conn.connector)
```

(Read more about [context managers here](https://docs.python.org/3/library/contextlib.html). You can also make use of `contextlib` in the Python standard library for more context manager fun!)

If you're ready to put your code into production, you may need a number of distinct database connectors, and consequently find that using a context manager is overly wordy. A great alternative is a decorator, which is a magical piece of code that modifies functions as opposed to objects. See below for our earlier example reconfigured as a decorator:

```python
import cx_Oracle
import logging

def db_connector(func):
    def with_connection_(*args, **kwargs):
        conn_str = conn_str
        cnn = cx_Oracle.connect(conn_str)
        try:
            rv = func(cnn, *args, **kwargs)
        except Exception:
            cnn.rollback()
            logging.error("Database connection error")
            raise
        else:
            cnn.commit()
        finally:
            cnn.close()
        return rv
    return with_connection_
```

Decorators allow you to change or extend a function's behavior without really changing the function itself. You just decorate a function with one line of code above its definition to give it a whole new capability.

### How It Works

When designing a decorator, you need to have an inner function and a decorator function. Here, `db_connector` is the decorator function, and `with_connection_` is the inner function that's only defined within the decorator. When calling this decorator, the decorated function `func` ultimately points to the decorator's inner function, `with_connection_`. This inner function then wraps the decorated function `func` by doing the following:

1. **Accepting an arbitrary number of positional and keyword arguments**, allowing the decorated function to operate normally: `with_connection_(*args, **kwargs)`

2. **Passing the database connector to the decorated function**: `func(cnn, *args, **kwargs)`

3. **Making sure the database connector is handled properly** by Python's try/except syntax, like we talked about earlier in the post.

With this in place, when you want to alter an existing function to use our database connector, you can drop in a decorator just above it like so:

```python
@db_connector
def do_some_job(cnn, arg1, arg2=None):
    cur = cnn.cursor()
    cur.execute(SQL, (arg1, arg2)) # or something else

do_some_job(arg1, arg2)
```

The `@` symbol syntax basically (though not literally) equates to the following:

```python
do_some_job(arg1, arg2) = db_connector(do_some_job(arg1, arg2))
```

This decorator creates a database connection and passes it along to the function below it. It basically does the job of a context manager, but in an even simpler way. There are [other really interesting ways to write decorators](https://docs.python.org/3/glossary.html#term-decorator) — see more here.

## Handling Credentials

So now you know how to set up a safe, easy database connection using context managers, decorators, and other best practices. But we're not done yet!

### The Problem

Here's a question: in any of the code above, how would you input your username and password to connect to the database? (Or rather, in this case, give the credentials to the aforementioned `conn_str`?)

One idea might be to have a `.py` file with all necessary credentials. This is certainly better than hard-coding your credentials into a `conn_str` variable in each script that uses a database connection. But hard-coding credentials, even if they're in a separate file, is never recommended.

Imagine that, in the beginning of your project, you used your personal password as credentials for a database connection. When it comes time to open-source your code, you might forget about the early days, but GitHub's commit history certainly didn't! Even absent any security concerns, if someone maintaining your project has to pore through all your code/data dictionaries to find the relevant snippets in which to update credentials, they will not be thrilled with your work.

**In short, hard-coding credentials is unwieldy and potentially dangerous.**

### The Solution: Environment Variables

On the other hand, there's another way to store your database credentials gracefully and safely: use **environment variables**. Whether you're using Windows or a Unix-based system, you can always set these variables and hide them within your system, and they're definitely more confidential than using scripts or configuration files.

In addition, when you're deploying into a container (e.g. Docker), you can easily set your environment variables and not worry about changing them between deployments. For example, when switching from a testing environment to a production environment, all the database configurations could be different, but you can change the configuration of the container without altering the underlying code. [Check here if you need a hint on how to change environment variables in your OS.](https://docs.python.org/3/library/os.html#os.environ)

### Implementation with Environment Variables

Here are our major previous code snippets with references to environment variables (see the `os.environ` bits):

#### Context Manager with Environment Variables

```python
import cx_Oracle
import logging
import os

class oracle_connection(object):
    """oracle db connection"""
    def __init__(self, connection_string=os.environ["CONN"]):
        self.connection_string = connection_string
        self.connector = None
    
    def __enter__(self):
        self.connector = cx_Oracle.connect(self.connection_string)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.connector.commit()
        else:
            self.connector.rollback()
        self.connector.close()
```

#### Decorator with Environment Variables

```python
import cx_Oracle
import logging
import os

def db_connector(func):
    def with_connection_(*args, **kwargs):
        conn_str = os.environ["CONN"]
        cnn = cx_Oracle.connect(conn_str)
        try:
            rv = func(cnn, *args, **kwargs)
        except Exception:
            cnn.rollback()
            logging.error("Database connection error")
            raise
        else:
            cnn.commit()
        finally:
            cnn.close()
        return rv
    return with_connection_
```

### Key Takeaways

- Use **environment variables** to store sensitive credentials instead of hard-coding them
- Environment variables can be easily configured in different environments (testing, production)
- When using Docker or other containerization, environment variables can be set at container startup
- This approach is more secure, maintainable, and follows security best practices
