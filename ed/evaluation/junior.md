# Junior Python Evaluation

## Goal
Assess foundational Python understanding for developers who are new to the language or still building core skills.

## Format
- 10-minute concept quiz
- 2 short coding tasks
- 1 debugging task
- 1 simple scenario

## Topics
- variables and types
- lists, dicts, and loops
- conditionals
- functions and return values
- basic file reading
- exceptions
- venv and pip basics

## Sample questions
1. What is the difference between a list and a tuple?
2. What does `len()` do?
3. How do you create a function in Python?
4. What is the purpose of a `return` statement?
5. What happens when a file does not exist and you try to open it?
6. What is the difference between `=` and `==`?

## Coding tasks

### Task 1: Sum even numbers
```python
def sum_even_numbers(numbers):
    total = 0
    for n in numbers:
        if n % 2 == 0:
            total += n
    return total
```

### Task 2: Read file and average numbers
```python
def average_from_file(path):
    total = 0
    count = 0
    with open(path, "r") as f:
        for line in f:
            total += float(line.strip())
            count += 1
    return total / count if count else 0
```

## Debugging task
Fix this broken snippet:
```python
name = "python"
print(name.upper())
```

## Scoring emphasis
- correctness
- understanding of core syntax
- ability to follow basic examples
- willingness to ask for clarification when needed

## Training priority
The junior version should surface gaps in:
- variables and assignment
- function basics
- loops and conditionals
- reading files
- environment setup
