# Python Team Assessment Answer Key

## Core quiz answers
1. List is mutable; tuple is immutable.
2. 8
3. KeyError
4. It sends a value back to the caller.
5. It isolates project dependencies.
6. `=` assigns; `==` compares.
7. 6
8. It catches and handles errors without crashing the program.
9. A module is a Python file or library of code.
10. `[2, 3]`

## Junior task sample
```python
def sum_even_numbers(numbers):
    total = 0
    for n in numbers:
        if n % 2 == 0:
            total += n
    return total
```

## Mid-level task sample
```python
def average_from_file(path):
    total = 0
    count = 0
    with open(path, "r") as f:
        for line in f:
            try:
                total += float(line.strip())
                count += 1
            except ValueError:
                continue
    return total / count if count else 0
```

## Senior task sample concept
A strong senior response should:
- validate input carefully
- ignore invalid rows instead of crashing
- use clear function boundaries
- document assumptions
- ensure edge cases are considered

## What to look for
- logic correctness
- edge-case handling
- readability
- good function names
- proper file handling
- robust validation
