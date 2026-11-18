# Python Training

## Day 1 (5/24/2022)

### Introduction to Python Basics

#### White Space & Formatting
- White space matters in Python
- Colons indicate scope breaks (e.g., classes, functions, control flow)
- Comments use `#`
- Shebang `#!` specifies an interpreter

#### Underscore Naming Conventions

| Pattern | Example | Meaning |
|---------|---------|---------|
| Single Leading Underscore | `_var` | Naming convention for internal use; not enforced but a hint to programmers |
| Single Trailing Underscore | `var_` | Used to avoid naming conflicts with Python keywords. Example: `class_` |
| Double Leading Underscore | `__var` | Triggers name mangling in class context; enforced by interpreter |
| Double Leading & Trailing | `__var__` | Special methods defined by Python; avoid for own attributes |
| Single Underscore | `_` | Temporary/insignificant variables ("don't care") or last REPL result |

```python
for _ in range(32):
    print("do something")
```

### Module Importing

```python
import x                    # Import module x
from y import x             # Import x from module y
import x as y               # Rename module on import
```

### Variables

- All variables are **objects** stored on the heap with a reference on the stack
- **Reference Count**: tracks how many references exist to an object
- When reference count reaches 0, garbage collector deallocates memory
- Two variables with same value point to the same object
- Changing a variable's value creates a new object in memory

#### Types

Common types: `str`, `int`, `float`, `tuple`, `list`, `dict`, etc.

Python is not strongly typed but you can indicate:
```python
x: str              # Variable x expected to be string
def func() -> str:  # Function returns string
```

- `type()`: Check current type of a variable
- `input()`: Request console input
- Type casting: `int("5")`, `str(42)`, etc.

#### Variable Scope: LEGB

1. **Local** - Inside current function
2. **Enclosing** - In enclosing function (nested functions)
3. **Global** - Module level
4. **Built-in** - Python's built-in namespace

### Strings

- F-strings (recommended): `f"hello {variable}"`
- Use backslashes for literals: `\"` for quotes
- Case methods: `upper()`, `lower()`, `title()`, `capitalize()`
- String slicing: `string[start:end]` (inclusive:exclusive, can blank for beginning/end, `-x` for backward)

### Numbers

- `int` and `float` types
- Size limited only by available memory
- Math operators: `+`, `-`, `*`, `**` (power), `/`, `//` (floor division), `%` (modulus)

```python
answer = 6**2       # 6 squared = 36
result = 10 // 3    # Floor division = 3
remainder = 10 % 3  # Modulus = 1
```

### Collections

#### Lists `[]`
- 0-based indexing
- Mutable (can be changed)
- Methods: `append()`, `extend()`, `index()`, `max()`, `min()`, `len()`, `clear()`, `insert()`, `count()`, `pop()`, `remove()`, `reverse()`, `copy()`
- Keyword: `in` to search

```python
my_list = [1, 2, 3, 4, 5]
my_list[1:3]        # [2, 3]
my_list[-3:]        # Last three elements
my_list.append(6)   # Add element
```

#### Tuples `()` or unpacking
- 0-based indexing
- Immutable (cannot be changed)
- Unpacking: `variable, variable = value, value`

#### Dictionaries `{}`
- Key-value pairs
- Keys must be hashable (no lists, but tuples or strings work)
- Methods: `.keys()`, `.values()`, `.items()`

```python
my_dict = {"name": "John", "age": 30}
my_dict["name"]          # Access
my_dict["city"] = "NYC"  # Add/update
my_dict.get("age")       # Safe access
```

### Operators

#### Relational Operators
- `==`, `!=`, `<`, `>`, `<=`, `>=`

### Control Flow

#### If/Elif/Else
```python
if x == 7:
    print("x is 7")
elif x > 7:
    print(f"x is bigger than 7: {x}")
else:
    print("x is not 7 or greater")
```

#### While Loop
```python
x = 6
while x > 0:
    print(x)
    x -= 1
    # continue or break
```

#### For Loop
```python
for variable in collection:
    # do something
    
for i in range(10):
    print(i)
```

### Day 1 Assignment: Voter Registration Validator

Create a console application that:
1. Asks for at least three things (one non-string)
2. Display at least one user value in a string
3. Perform a math function
4. Include at least one control flow statement

**Example Solution:**
```python
from datetime import date

print("Hello, and welcome to the voter registration validator")
while True:
    name = input("What is your name? ")
    print(f"Welcome {name}")
    yr = int(input("In which year were you born? "))
    now = date.today().year
    age = now - yr
    print(f"You are {age} years old")
    if age >= 18:
        print("You can vote!")
    else:
        print("You must be 18 to vote")
```

---

## Day 2 (6/7/2022)

### String Methods

#### Case Conversion
```python
a = "hello world"
a.title()       # "Hello World" (first letter of each word)
a.capitalize()  # "Hello world" (first letter of first word)
a.upper()       # "HELLO WORLD" (all uppercase)
a.lower()       # "hello world" (all lowercase)
```

#### F-String Magic
```python
a = "hello world"
print(f"My message: {a}")       # My message: hello world
print(f"{a=}")                  # a='hello world'
print(f"{5+3=}")                # 5+3=8
```

### Advanced Collections

#### DefaultDict
```python
from collections import defaultdict

def def_value():
    return "Not Present"

d = defaultdict(def_value)
d["a"] = 1
d["b"] = 2
print(d["c"])  # "Not Present" (auto-created with default value)
```

#### Counter
Returns count of unique characters/elements

```python
from collections import Counter

a = "hello world"
Counter(a)
# Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

b = ["hello", "world", "World", "world"]
Counter(b)
# Counter({'world': 2, 'hello': 1, 'World': 1})
```

#### Set
Collection of unique items (removes duplicates)

```python
b = ["hello", "world", "World", "world"]
set(b)  # {'hello', 'World', 'world'}
```

### JSON Handling

**Important:** JSON uses `true/false` (lowercase), Python uses `True/False` (capitalized)

### Sorting

```python
c = ["apple", "cat", "baby"]
c.sort()                    # Sorts in-place
print(c)                    # ['apple', 'baby', 'cat']

sorted(c, reverse=True)     # Returns new sorted list
# ['cat', 'baby', 'apple']
```

### Exception Handling (Try/Catch)

Traps errors and prevents crashes while providing helpful error messages.

```python
try:
    # code that might fail
    result = int(input("Enter a number: "))
except ValueError as err:
    print(f"Invalid input: {err}")
except Exception as err:
    print(f"Unexpected error: {err}")
```

**Re-raising Exceptions:**
When catching and re-raising, use `from` keyword to preserve the original exception:

```python
try:
    # code
except Exception as err:
    raise Exception(f"Error in happy function: {err}") from err
```

### Print & Input

```python
print("hello world")           # Output to console
a = input("What's your name? ")  # Receive input (always returns string)
```

### Conditionals

#### Ternary Operator
```python
x = 4 if b > 8 else 9  # x is 4 if b > 8, otherwise 9
# Read aloud: "x will be 4 if b is greater than 8, otherwise 9"
```

### Day 2 Assignment: Age Calculator

Create a console application that:
- Asks user for birthday in MM/DD/YYYY format
- Correctly calculates age regardless of current year
- Handles errors gracefully (doesn't crash)

**Example Solution:**
```python
from datetime import date
from dateutil import relativedelta

while True:
    try:
        bday_raw = input("What is your birthday (MM/DD/YYYY)? ")
        bday_array = bday_raw.split("/")
        bday = date(int(bday_array[2]), int(bday_array[0]), int(bday_array[1]))
    except ValueError as err:
        print(f"Please enter your birthday in MM/DD/YYYY format. {err}")
        continue
    except IndexError as err:
        print(f"Please enter your birthday in MM/DD/YYYY format. {err}")
        continue
    
    curr_date = date.today()
    date_diff = relativedelta.relativedelta(curr_date, bday)
    age = date_diff.years
    print(f"You are {age} years old")
    break
```

---

## Day 3 (Assignment: Tic-Tac-Toe Game)

Build a tic-tac-toe game without looking up online (but ask for help if needed).
- First player is "X"
- Second player is "O"
- Track wins across multiple games

**Example Solution:**

See [examples.py](../../../learning_game/tic-tac-toe/examples.py) for complete implementation.

---

## Functions

### Parameters

Best practice: Declare variable types on parameters and return values

```python
def my_function(age: str, address: str, name: str) -> int:
    """This function allows a lookup of a customer account
    
    Args:
        age (str): Age of the customer
        name (str): Name of the customer
    
    Returns:
        int: id of the customer account
    """
    pass
```

### Named Parameters

When calling functions with named parameters, order doesn't matter:

```python
my_function(name="John", age="30", address="123 Main St")
```

### Docstrings

Use docstrings for intellisense and auto-documentation. Use VS extension **autoDocstring - python** for easy creation.

### Inspecting Functions

Use the `inspect` module to get function information:

```python
import inspect

def fun(a, b):
    return a**b

print(inspect.signature(fun))
# Output: (a, b)

print(inspect.getargspec(fun))
# Output: ArgSpec(args=['a', 'b'], varargs=None, keywords=None, defaults=None)
```

---

## Anonymous Functions (Lambdas)

Single expression function without name.

```python
lambda x: x              # Identity function
lambda x: x + 1          # Add one
lambda x, y: x + y       # Multiple parameters
```

### Lambda Syntax
- Keyword: `lambda`
- Bound variable(s): `x` (argument)
- Body: expression after colon

### Type Hints in Lambdas
```python
lambda first: str, last: str: first.title() + " " + last.title() -> str
```

### Invoking Lambdas

#### IIFE (Immediately Invoked Function Expression)
```python
(lambda x, y: x + y)(2, 3)  # Returns 5
```

#### Named Lambda
```python
add_one = lambda x: x + 1
add_one(2)  # Returns 3
```

#### Using Underscore in REPL
```python
lambda x, y: x + y
_(1, 2)  # Returns 3
```

---

## Higher-Order Functions

### Filter

Takes a predicate and iterable, returns elements satisfying the predicate.

```python
even = lambda x: x % 2 == 0
list(filter(even, range(11)))
# [0, 2, 4, 6, 8, 10]
```

### Map

Takes a function and iterable, applies function to each element.

```python
list(map(lambda x: x.capitalize(), ['cat', 'dog', 'cow']))
# ['Cat', 'Dog', 'Cow']
```

### Reduce

Since Python 3, in `functools` module. Accumulates values by applying function to elements.

```python
import functools

pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
functools.reduce(lambda acc, pair: acc + pair[0], pairs, 0)
# 6 (sum of first elements)
```

---

## Decorators

Function that modifies or enhances another function or class without changing it.

```python
def trace(f):
    def wrap(*args, **kwargs):
        print(f"[TRACE] func: {f.__name__}, args: {args}, kwargs: {kwargs}")
        return f(*args, **kwargs)
    return wrap

@trace
def add_two(x):
    return x + 2

add_two(3)
# Output:
# [TRACE] func: add_two, args: (3,), kwargs: {}
# 5
```

Decorators can also be applied to lambdas:
```python
print((trace(lambda x: x ** 2))(3))
```

---

## Future Topics

- Functions continued: args, kwargs, default values, first-class citizens
- Classes and scope
- `self` keyword
- Advanced list operations
- Working with files: read, write, append
- JSON handling
- Built-in modules: math, statistics, os, sys
- Enums and data types
- Arguments with argparse
- VS Code magic/shortcuts
- Database operations
- Git and version control
- Debugging techniques
- Virtual environments
- Python packages and pip/requirements.txt
- Advanced keywords and dunders
- Memory management and default values

---

## Memory Management

### Garbage Collection

Python handles memory management automatically using:
- **Reference Counting Algorithm**: Counts references to each object
- When reference count = 0, garbage collector deallocates memory

### Object Storage

Everything in Python is an object:
- **Heap**: Stores objects (data)
- **Stack**: Stores references to objects and function calls

### Example Memory Layout
```python
x = 5
y = 6
a = 5
def func(a, b):
    c = a + b
    return c

# Stack contains: func, a, b, c, x, y
# Heap contains:
#   - 5 (referenced by x and a)
#   - 6 (referenced by y)
#   - 11 (result of func, temporarily referenced by c)
```

### Getting Memory Information

```python
id(variable)        # Returns unique identity (memory address)
hex(id(variable))   # Convert to hexadecimal

# Get object from memory address
import ctypes
a = ctypes.cast(x, ctypes.py_object).value
```

---

## Argparse Notes (Jan 11, 2023)

**Important:** `default` vs `const` keyword behavior:
- `default`: Used when argument is NOT provided
- `const`: Used when flag IS provided WITHOUT a value

```python
parser.add_argument("--debug", help="Debug", const=1, default=5)
```

Results:
- `python script.py` → debug = 5 (from default)
- `python script.py --debug` → debug = 1 (from const)
- `python script.py --debug 2` → debug = 2 (user-provided)

---

## Null-Coalescing Operations

Short null checking during assignment.

```python
# C# style
s = None
other = s ?? "some default value"

# Python equivalent
other = s if s is not None else "default value"
```

---

## Traceback

Capture and print stack traces for debugging:

```python
import traceback

try:
    # code
except Exception:
    traceback.print_exc()  # Print full stack trace
```

---

## Unit Testing

Video resources:
- [Session 1](https://fmr.zoom.us/rec/share/OGHuPBR0gs7LOM39Q0Bq2VoRnaOrTkXYmdRkuJl-HgoyCVPdm4FKcI3ml_OM5FJM.YPuum51I491GlJ5S) (Passcode: qC6^x8w?)
- [Session 2](https://fmr.zoom.us/rec/share/0MBwMBm_WUi4Mt_LPheR5u87Z4dZLlxzzDvxHsBDv2K7foxTKmspco9GYuyQU1cy.RMW4wiTmnf6ElM16) (Passcode: qK0X&k+c)
