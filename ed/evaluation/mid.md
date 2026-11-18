# Mid-Level Python Evaluation

## Goal
Measure practical Python skills for developers who can code but are still developing production-ready habits.

## Format
- 10-minute concept quiz
- 2 coding exercises
- 1 debugging task
- 1 scenario-based challenge

## Topics
- functions with return values
- dictionaries and lists
- file I/O and parsing
- exceptions and validation
- classes and object usage
- basic testing mindset
- working with formatted data

## Sample questions
1. What is the difference between a list comprehension and a loop?
2. Why is `with open(...)` preferred over `open(...)` without a context manager?
3. What is a dictionary key used for?
4. How can you validate user input before processing it?
5. What is a simple way to handle invalid data in a loop?
6. What is the benefit of writing tests for a function?

## Coding tasks

### Task 1: Average values from a file
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

### Task 2: Build a dictionary from names
```python
def name_lengths(names):
    result = {}
    for name in names:
        result[name] = len(name)
    return result
```

## Debugging task
Fix this function so it handles empty strings and invalid values gracefully:
```python
def normalize_name(value):
    return value.strip().title()
```

## Scenario challenge
Create a function that reads a list of records and removes any invalid row before saving a cleaned result.

## Scoring emphasis
- correctness under edge cases
- readability and naming
- use of validation
- clean handling of file operations

## Training priority
The mid-level version should surface gaps in:
- data validation
- reusable functions
- handling edge cases
- file parsing
- working with structured data
