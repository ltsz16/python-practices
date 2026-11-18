# Python
[TIOBE Index](https://www.tiobe.com/tiobe-index)

## Table of Contents
- [Getting Started](#getting-started)
- [Python Resources](#python-resources)
- [Overview](#overview)
- [What is it](#what-is-it)
- [Setup](#setup)
- [Env](#env)
- [Basics](#basics)
- [Virtual Environments](#virtual-environments)
- [Testing](#testing)
- [Debugging](#debugging)
- [Popular Libraries & Frameworks](#popular-libraries--frameworks)
- [Additional Topics](#additional-topics)
- [Data](#data)
- [Advanced Topics](#advanced-topics)
- [New since Python 3.10](#new-since-python-310)
- [Special Notes](#special-notes)
- [External Resources](#external-resources)
---

## Getting Started
Install all the following
[Visual Studio Code](https://code.visualstudio.com/)  
[Git](https://git-scm.com/downloads)  
[Get Python](https://www.python.org/downloads/)  
[VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)  
[VS Code Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack)

## Git
Using Github

## Python Resources
[3.12.0 Documentation (python.org)](https://docs.python.org/3/)  
[The Python Tutorial — Python 3.12.0 documentation](https://docs.python.org/3/tutorial/index.html)  
[The Python Language Reference — Python 3.12.0 documentation](https://docs.python.org/3/reference/index.html)  
[The Python Standard Library — Python 3.12.0 documentation](https://docs.python.org/3/library/index.html)  
[PEP 0 – Index of Python Enhancement Proposals (PEPs) | peps.python.org](https://peps.python.org/)  
[PyPI · The Python Package Index](https://pypi.org/)

## Overview
* High-level
* Dynamic
* Object-oriented
* Cross-platform
* Format matters
* Not strongly typed
* * Int not limited in size (high percision up to size of memory)

## What is it
* Standards: PEP (e.g. PEP 8)
* It is a spec and implementations of that spec (part installed)
These two are released *full* to the spec:
* * CPython (aka Python, classic python)
* * PyPy 
* IPython (interactive environent, i.e. Jupyter Notebook)
* Anaconda (includes preconfigred extenson modules based on Conda package manager)

octoverse.github.com

## Setup
1. Installing Python
2. [Setup environemnt](https://github.com/ericdboyd/vslive-2026-python-and-ai-day-1)

## Env
* python.exe
* Env Vars
* * PATH: where to find python.exe
* * PYTHONHOME: python installation dir with lib
* * PYHTONPATH: ist of modules
* * PYTHONSTARTUP: Pytho source to execute on start of interpreter
* REPL (Read, Evaluate, Print, Loop)
* * Interactive session
* PIP: Package manager
* * freeze > <file.txt>
* * install -r <file.txt>
* * install <module>
* * list

### Conventions
* requirements.txt (root and name)
* lower case and underscore nameing
* private is prefaced with an underscore

## Basics
* comments #
* Pass and ...
1. [Hello World](02 - Functions and Variables\demo)
2. Calculator
3. Functions
* colons
* indentation (convention says 4,, though not required)
* convention of a main function as entrypoint
* line order (top to end)
* returns
* typing on parameters and return
* *args **kwargs
4. Conditionals
* compare
* * and, or, <>=, startswith, contains, ==, !=, null
* * Booleans (True/False)
* if elif
* match
* * _ for else and | for multiple match
* line condensing:
    e.g. return True if n % 2 == 0 else False
5. Groups
* lists
* * sorted
* dictionaries
* sets
6. Loops
* while
* for
* * for in range
* incrementing/decrementing
* * += -+
**Project**: *Create tic-tac-toe*
7. Library importing
* import top level
* import function/class from module
* * be aware of overloading
* alias import
* import a custom file
* Looking up library docs
8. Random
* Number Generation: random, randint, uniform, randrange
* Sequence Operations: choice, choices, shuffle, sample
* Distribution: gauss, normalvariate, expovariate, betavariate, gammavariate, etc
* State control: seed, getstate, setstate
* secrets: secrets module
9. Arguments
* sys.argv
* argparse
* cowspeak
10. FIle IO
* open
* with
11. Error Handling & Exceptions
* try/except blocks
* except specific exceptions (ValueError, TypeError, etc.)
* else clause (runs if no exception)
* finally clause (always runs, for cleanup)
* raise custom exceptions
* Custom exception classes
* Exception hierarchy
**Project**: *Create a math testing app*

## Virtual Environments
* Isolate project dependencies
* venv module: `python -m venv env_name`
* Activate: `.\\env_name\\Scripts\\activate` (Windows)
* requirements.txt for dependency management

### Debugging
* pdb (Python debugger) - breakpoints, step through code
* print() debugging
* logging module for production debugging
* IDE debuggers (VS Code, PyCharm)
* traceback for error information

## Testing
* Types of testing
* * White box: unit, functional
* * Black Box: Smoke, load, acceptance
* unittest framework (built-in)
* pytest (popular third-party)
* Test discovery and test runners
* Assertions: `assert condition, "message"`
* Mock objects for testing without side effects

## Popular Libraries & Frameworks
* Data: numpy, pandas (data analysis)
* Web: Flask, Django (web applications)
* HTTP: requests (web scraping, API calls)
* JSON handling: json module
* Date/time: datetime module
* Regular expressions: re module
* REST: fastapi module

## Adtl Topics
### Comprehensions
* List comprehensions: `[x*2 for x in range(5)]`
* Dict comprehensions: `{k: v for k, v in list}`
* Set comprehensions: `{x for x in list if condition}`
* Generator expressions: `(x for x in range(1000))`

### Lambda Functions
* Anonymous single-expression functions
* Syntax: `lambda x: x * 2`
* Used with map(), filter(), sorted()
* Common in functional programming patterns

### Decorators
* Function/method wrappers that modify behavior
* @decorator syntax
* Built-in decorators: @property, @staticmethod, @classmethod
* Custom decorators with arguments
* functools.wraps for preserving metadata

### Context Managers
* with statement for resource management
* __enter__() and __exit__() dunders
* Automatic cleanup (files, connections, locks)
* Example: `with open(file) as f:`

### Generators & Iterators
* yield keyword for lazy evaluation
* Generators save memory (don't compute all values upfront)
* Iterator protocol: __iter__() and __next__()
* Use cases: large datasets, infinite sequences

### Type Hints
* Optional type annotations on parameters and returns
* Syntax: `def func(name: str, age: int) -> bool:`
* Type checking tools: mypy, Pylance
* Improves code readability and IDE support

### Async/Await (Asynchronous Programming)
* async/await keywords for concurrent code
* asyncio module for event loops
* Non-blocking I/O operations
* Useful for web scraping, API calls, networking

### Common Patterns
* Singleton pattern
* Factory pattern
* Observer pattern
* Builder pattern
* Context manager pattern
* Decorator pattern

### Performance & Optimization
* List comprehensions vs loops (faster)
* Generator expressions for memory efficiency
* functools.lru_cache for memoization
* Profiling tools: cProfile, timeit
* Avoiding common pitfalls: mutable default arguments

### Module Organization
* __name__ == "__main__" pattern
* Package structure and __init__.py
* Relative vs absolute imports
* Module reload (importlib.reload)

### Threading (Multithreading)
* threading module for concurrent execution
* Thread class: create and manage threads
* `Thread(target=func, args=(arg1,))` to create a thread
* `.start()` to begin thread execution
* `.join()` to wait for thread completion
* Global Interpreter Lock (GIL) — only one thread executes Python bytecode at a time
* Useful for I/O-bound tasks (file, network, database)
* NOT useful for CPU-bound tasks (use multiprocessing instead)
* Race conditions when multiple threads access same data
* Lock/Mutex for synchronization: `threading.Lock()`
* `with lock:` for safe access to shared resources
* Thread-safe queues: `queue.Queue()`
* Daemon threads: background threads that exit when main program exits

### Regex (Regular Expressions)
* re module for pattern matching
* Pattern syntax: . (any char), * (0+), + (1+), ? (0-1), [] (character class)
* ^ (start), $ (end), | (or), () (group)
* re.search() — Find first match
* re.findall() — Find all matches
* re.sub() — Replace matches
* re.match() — Match at start of string
* Flags: re.IGNORECASE, re.MULTILINE, re.DOTALL
* Grouping and backreferences for extracting data

### OOP (Object-Oriented Programming)
[example](10 - OOP\demo\student20.py)
[example-inheritance](10 - OOP\demo\wizard.py)
* Classes and objects
* Attributes (data) and methods (functions)
* Constructor: __init__()
* self parameter (reference to instance)
* Instance vs class variables
* Inheritance (subclasses extend parent classes)
* super() to call parent methods
* Method overriding
* Private variables (prefix with _)
* Encapsulation and abstraction
* Polymorphism (objects of different classes respond to same method name)
* Abstract base classes (ABC) for interfaces
* Dunder methods for custom behavior

### math
* math module for mathematical functions
* Constants: math.pi, math.e, math.tau, math.inf
* Basic: math.sqrt(), math.pow(), math.fabs()
* Trigonometry: math.sin(), math.cos(), math.tan()
* Logarithms: math.log(), math.log10(), math.exp()
* Rounding: math.floor(), math.ceil()
* Combinatorics: math.factorial(), math.gcd()
* Degrees/radians: math.degrees(), math.radians()

## Data
### Working with Data Files
* CSV files
  * csv module: `csv.reader()`, `csv.writer()`, `csv.DictReader()`, `csv.DictWriter()`
  * pandas: `pd.read_csv()`, `df.to_csv()`
* JSON files
  * json module: `json.load()`, `json.dump()`, `json.loads()`, `json.dumps()`
  * Parsing and serializing Python objects
* YAML files
  * PyYAML library: `yaml.safe_load()`, `yaml.dump()`
  * Configuration file format
* Excel files
  * openpyxl or pandas for reading/writing .xlsx files
* Pickle (Python serialization)
  * `pickle.dump()`, `pickle.load()` for saving Python objects

### Data Structures for Data Handling
* DataFrames (pandas)
  * Tabular data with rows and columns
  * `pd.DataFrame()` constructor
  * Indexing, slicing, filtering
* Dictionaries for structured data
* Named tuples for lightweight data records
* Data classes (@dataclass) for clean data representation
* Series (pandas) for 1D labeled arrays
* Pydantic

### Data Validation & Cleaning
* Type checking and validation
* Handling missing/null values (np.nan, pd.isna())
* Removing duplicates (pd.drop_duplicates())
* Data type conversion
* Input validation with try/except or libraries like pydantic
* Regex for data validation and extraction

### Data Transformation
* Filtering data (boolean indexing, `.query()`)
* Mapping/transforming values (`.apply()`, `.map()`)
* Grouping and aggregation (`.groupby()`, `.agg()`)
* Sorting and ranking
* Pivoting data (`.pivot()`, `.pivot_table()`)
* Merging and joining datasets (`.merge()`, `.join()`)

### Working with Databases
* SQLite (built-in, simple file-based)
* Connection and cursors
* SQL queries: SELECT, INSERT, UPDATE, DELETE
* Object-relational mapping (ORM) with SQLAlchemy
* Query builders and data access patterns

### Data Analysis Basics
* Descriptive statistics (mean, median, std, min, max)
* Correlation and covariance
* Grouping and cross-tabulation
* Time-series data handling

### Data Visualization
* Matplotlib, Plotly, Seaborn

## Advanced topics
### Web Scraping
* BeautifulSoup, Scrapy

### Scientific Computing
* NumPy, SciPy, Pandas

### [AI](python_ai_overview.md)
* Linear Regression
* Random Forest
* Decision Trees
* Transformers

### [Other](python_adv_overview.md)
* Multiprocessing for CPU-bound parallelism (vs threading for I/O-bound)
* Descriptors and properties (advanced attribute access control)
* Metaclasses (classes that create classes)
* Slots (__slots__) for memory optimization
* Context variables for async context management
* Named tuples (collections.namedtuple, typing.NamedTuple)
* Pickle/serialization for object persistence
* Networking (sockets, client-server communication)
* Database ORMs (SQLAlchemy)
* Web frameworks beyond Flask/Django (FastAPI, Starlette)
* Machine learning frameworks (scikit-learn, TensorFlow, PyTorch, JAX)
* Data visualization (Matplotlib, Plotly, Seaborn)
* Package development and distribution (setuptools, poetry, pyproject.toml)
* Garbage collection and memory management
* Bytecode and CPython internals
* C extensions and ctypes for performance-critical code

## New since Python 3.10
### Python 3.10
* **Structural Pattern Matching** — match/case statement for complex control flow
  * `match value:`
  * `case pattern: ...`
  * Supports destructuring, guards, wildcards
* **Type Hints Improvements**
  * Union types with `|` operator: `int | str` instead of `Union[int, str]`
  * Type hints on function parameters and returns more concise
* **Parenthesized Context Managers** — multiple `with` statements on one line
* **New Built-in Type Hints** — `types.UnionType`

### Python 3.11
* **Exception Groups** — PEP 654 for handling multiple exceptions
  * `except* ExceptionType:` for grouping related exceptions
  * `BaseExceptionGroup` and `ExceptionGroup`
* **Async Task Groups** — `asyncio.TaskGroup()` for managing async tasks
* **StrEnum** — String-based enums (enum.StrEnum)
* **TypedDict** — Define dict structure with type hints
* **Performance Improvements** — 10-60% faster than 3.10
* **Tomllib** — Built-in TOML parsing

### Python 3.12
* **Type Parameter Syntax** — PEP 695 generics without `TypeVar`
  * `def func[T](x: T) -> T:` instead of `T = TypeVar('T')`
* **Override Decorator** — `@override` to mark overridden methods
* **f-string Improvements** — Debug syntax `f"{variable=}"` shows variable name and value
* **Per-Interpreter GIL** — Experimental sub-interpreters with separate GILs
* **Improved Error Messages** — Better suggestions for typos and mistakes

### Python 3.13
* **JIT Compilation (Experimental)** — Just-in-time compilation for better performance
* **PEP 692 TypedDict Unpacking** — `**TypedDict` in function signatures
* **Improved Async** — Better async context manager support
* **Deprecations Warnings** — Removal of deprecated features
* **Performance Optimizations** — Further improvements to execution speed

## Special Notes
### Variables
* Not strongly typed
* float is different: it is usually IEEE 754 binary64, which is the same as a C/Java double.
* * float precision (~ 15-17 decimal digits)
| Type | Size limit | Notes |
| --- | --- | --- |
| int | No practical limit; grows with memory | Arbitrary precision integers |
| float | Fixed 64-bit on most systems | IEEE 754 binary64, like a C/Java double |
| float precision | ~ 15-17 decimal digits | Not exact for values like 0.1 and 0.2 |
| float range | Max about 1.8e308 | Min positive about 5e-324 |

* Python variables are not limited the same way as fixed-size variables in many other languages.
* `int` can grow as large as memory allows.
* `float` is still limited by range and precision.

## External Resources
[HarvardX edX Courses](https://www.edx.org/school/harvardx)  
[CS50 Python Course](https://www.edx.org/learn/python/harvard-university-cs50-s-introduction-to-programming-with-python)  
[Python 3 Tutorial - Learn Python 3 (geeksforgeeks.org)](https://www.geeksforgeeks.org/python3-tutorial/)  
[Introduction To Python – GeeksforGeeks](https://www.geeksforgeeks.org/introduction-to-python/)  
[Get Started Tutorial for Python in Visual Studio Code](https://code.visualstudio.com/docs/python/python-tutorial)  
[Python for beginners - Training | Microsoft Learn](https://learn.microsoft.com/en-us/training/paths/beginner-python/)  
[Python Tutorial (w3schools.com)](https://www.w3schools.com/python/)  
[Learn Python - Free Interactive Python Tutorial](https://www.learnpython.org/) 