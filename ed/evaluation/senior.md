# Senior Python Evaluation

## Goal
Assess practical Python depth, design judgment, maintainability, and the ability to write clean, robust solutions for real engineering work.

## Format
- 15-minute concept check
- 3 code challenges
- 1 debugging exercise
- 1 design/scenario discussion

## Topics
- functions, decorators, and closures
- comprehensions and generators
- exceptions and validation
- classes and data modeling
- standard library usage
- testing mindset
- file processing and data pipelines
- performance awareness
- clean API and module design

## Sample questions
1. What is the difference between a list comprehension and a generator expression?
2. When would you use `dataclasses` instead of a plain class?
3. How do you handle invalid input in a public function?
4. What is the value of using `try/except` at the correct boundary?
5. Why is a `with open()` block preferred for file handling?
6. What is the tradeoff between using a dict and a list for lookup-heavy workloads?

## Coding tasks

### Task 1: Data normalization
Write a function that normalizes a list of records and filters out invalid entries.

### Task 2: CSV processing
Read a CSV file and return the top N rows by a given field.

### Task 3: Class design
Design a simple `Student` or `Order` model with validation and a summary method.

## Debugging task
Find and fix logic issues in a function that calculates the median, ignores invalid data, and handles empty input correctly.

## Scenario discussion
Ask the candidate to explain:
- how they would validate inputs
- how they would structure a reusable module
- how they would write tests for edge cases
- how they would improve performance on larger data sets

## Scoring emphasis
- correctness under edge cases
- readability and maintainability
- proper exception handling
- ability to explain tradeoffs
- good naming and modularity

## Training priority
The senior version should surface gaps in:
- design quality
- exception boundaries
- data structure choices
- performance and scaling awareness
- testability and refactoring skills
