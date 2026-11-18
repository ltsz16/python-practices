# Unit Tests

## Table of Contents

1. [Code Coverage](#code-coverage)
2. [Code Snippets](#code-snippets)
3. [Mock Dependencies](#mock-dependencies)
4. [Test with Expected Error](#test-with-expected-error)
5. [Skipping/Flaky Tests](#skippingflaky-tests)
6. [Mock REST Endpoints](#mock-rest-endpoints)
7. [Mock Argparse](#mock-argparse)
8. [Mock os.system Calls](#mock-ossystem-calls)
9. [Mock Library Import](#mock-library-import)
10. [Mock Sub-Function Calls](#mock-sub-function-calls)
11. [Mocking File Reading](#mocking-file-reading)
12. [Testing Async Functions](#testing-async-functions)
13. [Read Console Output](#read-console-output)

## Code Coverage

To include line coverage in your tests, you'll need the `coverage` package.

### Installation

```bash
pip install coverage
```

### Running Coverage with Pytest

If your tests are in a folder called `selfservice/unit_test`, run this command:

```bash
coverage run -m pytest -v selfservice/unit_test/ --junitxml=out_report.xml
coverage report
```

### Generate Report Formats

You also have the option to create XML or HTML versions of the report:

```bash
coverage xml -o coverage.xml
coverage html
```

This creates an HTML report in the `htmlcov/` directory that you can open in a browser for detailed coverage visualization.

## Code Snippets

This section contains practical code examples for various testing scenarios throughout this document.

## Mock Dependencies

When you need to test a file that imports a library which tries to make a database connection immediately upon import, you'll encounter import failures in a unit test environment. The solution is to mock these dependencies **before** importing the module.

### Example: Testing with Database Imports

Consider the following file you want to test (`audit/audit_logging.py`):

```python
from lib.logging import LoggingSystem
from sqlalchemy.orm import Session as db_session
import pymysql
from database import Base, Engine
from typing import Optional, List, Union
from pydantic import BaseModel, ValidationError, validator
from datetime import datetime
from common import get_db_info
from audit.schema import DBAuditReport, BacktestingAuditLog
from database import DCC_DB

log_sys = LoggingSystem()
logger = log_sys.logger
db = DCC_DB()

class AuditLogger():
    def function_to_test(self):
        # mycode
        pass
```

### Test with Mocked Dependencies

```python
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
import sys
import pathlib
import os

sys.path.insert(1, f'{pathlib.Path(__file__).parent.absolute()}/..')

# Mock dependencies BEFORE any imports from audit_logging
with patch('audit.database.get_db_info') as mock_get_db_info, \
     patch('audit.audit_logging.DCC_DB') as mock_dcc_db, \
     patch('audit.audit_logging.LoggingSystem') as mock_logging:
    
    mock_get_db_info.return_value = {
        'host': 'localhost',
        'user': 'test_user',
        'password': 'test_pass',
        'database': 'test_db'
    }
    mock_dcc_db.return_value = MagicMock()
    mock_logging.return_value.logger = MagicMock()
    
    # Now import after mocking
    from audit.audit_logging import AuditLogger, APIRequestData

@pytest.fixture
def audit_logger():
    return AuditLogger()
```

**Key Point:** Mock dependencies must be established in a context manager **before** importing the module. This prevents initialization errors when the module loads.

## Test with Expected Error

When testing functions that raise exceptions, use `pytest.raises()` to verify the correct exception is raised with the expected message.

### Function to Test

```python
def my_function(first_name, last_name):
    if first_name == "Tom":
        raise NameException("wrong name")
    return first_name + " " + last_name
```

### Test Cases for Exceptions

```python
import pytest

def test_my_function_format_error():
    with pytest.raises(Exception) as err:
        resp = my_function("Luke", 2)
    assert 'can only concatenate str (not "int") to str' in str(err)

def test_my_function_wrong_name():
    with pytest.raises(NameException) as err:
        resp = my_function("Tom", "Smith")
    assert 'wrong name' in str(err)
```

**Pattern:** Use `pytest.raises()` as a context manager to catch and verify exceptions. Access the exception details through the `err` object.

## Skipping/Flaky Tests

Pytest provides decorators to handle tests that should be skipped or are known to be flaky.

### Skipping Tests

Always specify a reason when skipping a test:

```python
@pytest.mark.skip(reason="need to adjust call")
async def test_get_gov_gate_v2_response_exception_cft(mocker):
    url = get_v2_url("cft")
    responses.post(url=url, json={}, status=500)
    asynctest.mock.patch('lib.fid_oauth.get_token', return_value="thisismytoken")
```

### Marking Flaky Tests

For tests that are intermittently failing, use the `@flaky` decorator to retry:

```python
from flaky import flaky

@flaky(max_runs=3, min_passes=2)
def test_something_that_usually_passes():
    _double = 21
    result = get_result_from_flaky_doubler(_double)
    assertEqual(result, _double * 2, 'Result doubled incorrectly.')
```

**Behavior:** The test will run up to 3 times and must pass at least 2 of those runs to be considered passing.

## Mock REST Endpoints

When testing code that makes HTTP requests, use the `responses` library to mock HTTP endpoints.

### Function Making HTTP Calls

```python
def get_info(endpoint: str, payload: dict):
    response = http.post(url=endpoint, json=payload, timeout=120, verify=False)
    if response.status_code != 200:
        return("ERROR: There was an error during your attempted deploy.")
    response = http.get(url="http://myendpoint.com/name", param=payload, timeout=120, verify=False)
    if response.status_code != 200:
        return("ERROR: There was an error getting deploy status.")
    return "finished"
```

### Test with Mocked Endpoints

```python
import responses

@responses.activate
def test_get_info():
    # note that each endpoint being hit must be mocked
    sst_endpoint = "https://aps-selfservice-nonprod.fmr.com/v1/deployment/eks"
    second_endpoint = "http://myendpoint.com/name"
    
    # Earlier versions of the library
    responses.post(url=sst_endpoint, json={}, status=200)  # Mock POST call
    responses.get(url=second_endpoint, json={}, status=400)  # Mock GET call
    
    # Current version of the library
    responses.add(responses.GET, url=sst_endpoint, body={}, status=200)
    responses.add(responses.POST, url=sst_endpoint, json={}, status=400)
    
    resp = get_info(sst_endpoint, {})
    
    # We expect the second error message because we mocked that the first REST call
    # passed and the second one failed
    assert resp == "ERROR: There was an error getting deploy status"
```

**Important:** Each endpoint being hit in your function must be mocked individually using `responses.add()` or the older `responses.post()`/`responses.get()` methods.

## Mock Argparse

When testing code that uses `argparse` for command-line arguments, mock `sys.argv` in your tests.

### Function Using Argparse

```python
import argparse

def add_funcs():
    parser = argparse.ArgumentParser()
    parser.add_argument("age", type=int)
    args = parser.parse_args()
    print(str(args.age))
```

### Test Valid and Invalid Input

```python
import argparse
import sys
import pytest
from io import StringIO
from myfile import add_funcs

def test_valid_input(capsys):
    sys.argv = ["myfile.py", "16"]
    add_funcs()
    captured = capsys.readouterr()
    assert captured.out == "16"

def test_invalid_input(capsys):
    sys.argv = ["myfile.py", "happpy"]
    with pytest.raises(SystemExit):
        add_funcs()
    captured = capsys.readouterr()
    assert "invalid int value" in captured.err
```

**Pattern:** Set `sys.argv` with the script name and arguments, then capture output using `capsys.readouterr()`.

## Mock os.system Calls

When testing code that calls system commands via `os.system()`, use `patch` from `unittest.mock` to mock system calls.

### Function with System Calls

```python
import os

def my_function():
    os.system("some command")
    os.system("command for a second system in same function")
```

### Test with Mocked System Calls

```python
import myfile
import pytest
from unittest.mock import patch

def test_my_function():
    with patch('os.system') as mock_system_1:
        mock_system_1.return_value = 0  # you can set the return code
        
        # now mock second call
        with patch('os.system') as mock_system_2:
            mock_system_2.return_value = 1
            my_function()
            mock_system_1.assert_called_once_with("some command")
            mock_system_2.assert_called_once_with("command for a second system in same function")
```

**Note:** Each nested `patch` context manager can mock different return values or behaviors for successive calls.

## Mock Library Import

When testing code that imports external libraries that may not be available or shouldn't be called, use `monkeypatch` to mock library imports and their functions.

### Code Using External Library

```python
import my_library

def my_function():
    return my_library.its_function()
```

### Test with Mocked Library

```python
import pytest
from my_module import my_function

def test_missing_library(monkeypatch):
    # Mock the library's function using monkeypatch
    monkeypatch.setattr(__import__('my_library'), "its_function", lambda: "Function Mocked")
    
    # What is happening above:
    # __import__('my_library') dynamically imports the "my_library" module.
    # It is equivalent to: import my_library
    # 'its_function' is the attribute (in this case a function) within the
    # "my_library" module which you want to mock
    # lambda: "Function Mocked" - This lambda function acts as a replacement
    # for the original "its_function" and here will return a string
    # "Function Mocked" instead of executing the original function
    
    result = my_function()
    assert result == "Function Mocked"
```

## Mock Sub-Function Calls

When testing a function that calls other functions, mock the sub-function to control its behavior.

### Function Calling Another Function

```python
def verify_permissions(account_details: dict) -> str:
    account_settings = get_aws_account_settings(account_details['level'])
    if account_settings == "aws":
        return "valid account"
    return "account not supported"
```

### Test with Mocked Sub-Function

```python
import pytest

def test_verify_permissions_prod_invalid_cm(mocker):
    acnt1 = {"level": "prod", "id": "123456"}
    mocker.patch("get_aws_account_settings", return_value="aws")
    resp = verify_permissions(acnt1)
    assert resp == "valid account"
```

### Mocking Methods Inside a Class

If the sub-function is inside a class (using `self.function` pattern), use `mocker.patch.object`:

```python
def test_view_user_mock(mocker, ldap_tools):
    mocker.patch.object(ldap_tools, "ldap_list_user_groups",
        return_value=[
            "CN=Group1,OU=Test,DC=example,DC=com",
            "CN=Group2,OU=Test,DC=example,DC=com"
        ])
    result = ldap_tools.view_user("test_user")
    assert "test_user belongs to the following AD Groups" in result
    assert "Group1" in result
    assert "Group2" in result
```

**Pattern:** For class methods, use `mocker.patch.object(instance, "method_name", return_value=...)` to mock internal method calls.

## Mocking File Reading

When testing code that reads files, use `mock_open` to mock the file I/O without creating actual files.

### Function Reading a File

```python
def calc():
    try:
        with open("data.txt", "r") as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return "File not found"
```

### Test Successful File Read

```python
import pytest
from unittest.mock import mock_open
from code import calc

def test_calc(mocker):
    # create a mock for open function
    m_open = mock_open(read_data="hello world")
    # replace open function with the mock
    mocker.patch("builtins.open", m_open)
    result = calc()
    assert result == "hello world"
    # ensure open function called correctly
    m_open.assert_called_once_with("data.txt", "r")
```

### Test File Not Found Exception

```python
def test_calc_file_not_found(mocker):
    m_open = mock_open()
    m_open.side_effect = FileNotFoundError
    mocker.patch("builtins.open", m_open)
    result = calc()
    assert result == "File not found"
```

**Common side effects:**
- `FileNotFoundError`
- `PermissionError`
- Other I/O exceptions

## Testing Async Functions

To test async functions, mark the test as async and use the `@pytest.mark.asyncio` decorator.

### Async Function to Test

```python
async def my_async_function(name: str) -> str:
    print(name)
    return "function called"
```

### Test Async Function

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_my_async_function():
    # Notice that the test function must also be async
    result = await my_async_function("test")
    assert result == "function called"
```

**Requirements:**
- Install: `pip install pytest-asyncio`
- Use `@pytest.mark.asyncio` decorator on test function
- Make test function `async def` (not regular `def`)
- Use `await` to call async functions

## Read Console Output

When testing functions that print output to the console, use the `capsys` fixture to capture stdout and stderr.

### Function Printing to Console

```python
def my_function():
    print("Hello world")
```

### Test with Captured Output

```python
import pytest
from my_code import my_function

def test_my_function(capsys):
    my_function()
    captured = capsys.readouterr()  # Captures both stdout and stderr
    assert captured.out.strip() == "Hello world"
```

### Capturing Error Output

```python
def test_my_function_stderr(capsys):
    # Function that prints to stderr
    some_function_with_error()
    captured = capsys.readouterr()
    # To read error output use captured.err instead of captured.out
    assert "error message" in captured.err
```

**Fixture:** `capsys` is a built-in pytest fixture that captures `stdout` and `stderr`.

- `captured.out` — Standard output
- `captured.err` — Standard error (stderr)

## Summary

### Key Takeaways

- **Coverage:** Use `coverage` package for line coverage measurement
- **Mocking:** Use `unittest.mock`, `mocker` (pytest-mock), and `monkeypatch` for different mocking scenarios
- **Exceptions:** Use `pytest.raises()` to test exception handling
- **REST APIs:** Use `responses` library to mock HTTP endpoints
- **Files:** Use `mock_open` to mock file I/O operations
- **Async:** Use `@pytest.mark.asyncio` for testing coroutines
- **Output:** Use `capsys` fixture to capture console output
- **Skipping:** Use `@pytest.mark.skip()` with reasons, and `@flaky()` for unreliable tests

### Installation Commands

```bash
pip install pytest
pip install pytest-mock
pip install pytest-asyncio
pip install coverage
pip install responses
pip install flaky
```

These packages provide comprehensive mocking and testing capabilities for Python unit testing.
