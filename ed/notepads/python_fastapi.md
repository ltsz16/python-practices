# FastAPI

## Table of Contents
- [Setup](#setup)
- [VERB Examples](#verb-examples)
- [Passing Data](#passing-data)
- [Base FastAPI in Docker](#base-fastapi-in-docker)
- [Using a Backend Database](#using-a-backend-database)
- [Authentication](#authentication)
- [Custom FastAPI Exceptions](#custom-fastapi-exceptions)
- [When to use Async](#when-to-use-async)
- [Internal CRON](#internal-cron)

## Setup

### Install

```bash
python -m pip install "fastapi[all]"
```

You can run your app manually with:

```bash
uvicorn <file to run>:<name of the app to run> --reload
```

The `--reload` switch will reload the application automatically when you change the code.

**Example:** If my file was called `main.py`, I would execute:
```bash
uvicorn main:app --reload
```

### Documentation

- View the Swagger documentation (automatically generated) by calling `<base url>/docs`
- Alternative format at `<base url>/redoc`
- View the OpenAPI spec at `<base url>/openapi.json`

### Create a Basic App

```python
from fastapi import FastAPI, Response, Depends, Header

# Set the name of the app, here we call it "app" but it could be called something else
app = FastAPI(
    title="APS Self-Service Tool",
    version=common.get_app_version()
)

# The first item specifies the verb (e.g. GET, POST), next is the endpoint in the url.
# So this is <URL>/healthcheck. The tags represent the "category" of the endpoint in the Swagger
@app.get("/healthcheck", tags=["Default"])
async def healthcheck():
    # This will return a dictionary to the end-user when they call the endpoint
    return {"status": "healthy"}
```

## VERB Examples

### GET Request with Query Parameters

```python
@app.get("/authorize-deploy", tags=["Authorization"], summary="Verify that request to deploy can be allowed")
async def is_auth(app_id:str, aws_acnt:str, env:str, current_user: auth.TokenData = Security(auth.describe_current_user)):
    resp = await user_auth.is_authorized(application_id=app_id, user=current_user, aws_account=aws_acnt, environment=env)
    return json.dumps(dataclasses.asdict(resp))
```

### POST Request

```python
@app.post("/debug-level", tags=["Logs"])
def set_debug(debug_level:str, current_user: TokenData = Security(describe_current_user, scopes=["opsengr"])):
    log_sys.reset_logging_level(log_level_enum[debug_level.upper()])
    return {"Log Level": log_sys.loglevel}
```

## Passing Data

### Path Parameters

```python
@app.get("/course/{course_id}")
def my_course(course_id: int):
    return {"course_id": course_id}
```

### Query Parameters

For example: `/my/page/items/?page=1&limit=10&order=0`

```python
@app.get("/my/page/items/")
async def read_item(page: int = 0, limit: int = 0, skip: int = 1):
    return dummy_data[page*10: page*10 + limit: skip]
```

### Request Body

This call only works with POST requests, not GET.

```python
from pydantic import BaseModel

class MyItem(BaseModel):
    name: str
    info: str = None
    price: float
    qty: int

@app.post("/purchase/item/")
async def create_item(item: MyItem):
    return {"amount": item.qty*100, "success": True}
```

**Example curl call:**
```bash
curl -X POST "http://127.0.0.1:8000/purchase/item/" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"sample item\",\"info\":\"This is info for the item\",\"price\":40,\"qty\":2}"
```

### Form Data

This is sent different than JSON and is sent from HTML forms `<form></form>`.

You will need to install the python-multipart library:

```bash
python -m pip install python-multipart
```

```python
from fastapi import Form

# Takes a username and password as form fields
@app.post("/accounts/login/")
async def login_view(username: str = Form(...), password: str = Form(...)):
    return {"success": True}
```

## Base FastAPI in Docker

```dockerfile
FROM fcr.fmr.com/fmr-ap135646/selfservice-base:latest
LABEL maintainer="apscloudops@fmr.com"

# Create a local user
RUN useradd -u 8877 -m sst_user && \
    usermod -aG sudo sst_user

# Install required python libraries
ADD requirements.txt build/requirements.txt
RUN pip install -r build/requirements.txt

USER sst_user
WORKDIR /app

# Using script to create a self-signed SSL cert.
# This is a stand-alone script which exists in the code
ADD scripts/generate_selfcert.sh scripts/generate_selfcert.sh
RUN scripts/generate_selfcert.sh

# Copy over the source code to the docker container
COPY selfservice /app

# This is a note. You must still open the port when you run the image
EXPOSE 8000

CMD ["gunicorn", "main:app", "--timeout", "90", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0", "--keyfile=/tmp/aps.local.key", "--certfile=/tmp/aps.local.crt"]
```

## Using a Backend Database

### SQLModel Library

The `sqlmodel` library allows tables to be auto-created and data stored there. It's built on SQLAlchemy and Pydantic.

```bash
pip install sqlmodel
```

**Note:** If you are in developer mode with FastAPI running, you will want to stop it until you complete the code for creating your tables so that you don't get anything in a partially created state.

### Data Model

```python
from sqlmodel import SQLModel, Field

class CarInput(SQLModel):
    size: str
    fuel: str | None = "electric"
    doors: int
    transmission: str | None = "auto"

# Here we will create a table called Car which will have all the rows of CarInput 
# but also a unique id row too
class Trip(CarInput, table=True):  # this will create the table if it doesn't already exist
    id: int | None = Field(primary_key=True, default=None)
```

### DB Connection

```python
from fastapi import FastAPI
from sqlmodel import create_engine, SQLModel

app = FastAPI(title="My application")

# This is an example connecting to SQLite. Use appropriate DB connection string 
# (e.g. MySQL, MS SQL, etc)
engine = create_engine(
    "sqlite:///mydatabase.db",
    connect_args={
        "check_same_thread": False,  # needed for sqlite
        "echo": True  # log generated SQL
    }
)

@app.on_event("startup")  
# This will run on start-up but after all code loads, and therefore all model classes are known
def on_startup():
    # Will check if db exists and create if it doesn't
    # Will create tables for those set as table=True
    SQLModel.metadata.create_all(engine)
```

### CRUD Operations

```python
from sqlmodel import create_engine, SQLModel, Session, select

@app.post("/mydata")
def add_data(new_car: CarInput):
    # Create a data session importing the engine object exemplified in DB Connection above
    with Session(engine) as session:
        # Create instance since properties align and db relationships are kept
        new_car_load = DataInputModel.from_orm(new_car)
        session.add(new_car_load)
        session.commit()  # create new row
        session.refresh(CarInput)  # updates our object with the id which would have been created in the database
        return data_input

@app.get("/cars")
def get_car(size: str | None = None, doors: int | None = None) -> list:
    with Session(engine) as session:
        query = select(Car)  # creates a query object on the Car table
        # Create a query which can be one or both filters
        if size:
            query = query.where(Car.size == size)
        if doors:
            query = query.where(Car.doors >= doors)
        # Returns a results object. .all() asks all results to be converted as python objects
        return session.exec(query).all()
```

## Authentication

### Base Authentication - User Model

```python
from sqlmodel import SQLModel, Field, Relationship, Column, VARCHAR
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"])

class UserModel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column("username", VARCHAR, unique=True, index=True))
    password_hash: str = ""
    
    def set_password(self, password):
        """Setting the password sets the hash"""
        self.password_hash = pwd_context.hash(password)
    
    def verify_password(self, password):
        """Get hash and compare with db hash"""
        return pwd_context.verify(password, self.password_hash)

class UserOutput(SQLModel):
    id: int
    username: str
```

**Install password hashing:**
```bash
pip install "passlib[bcrypt]"
```

### HTTP Basic Authentication

```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException
from sqlmodel import Session, select
from starlette import status
from db import get_session
from schemas import UserOutput, User

security = HTTPBasic()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security), 
                      session: Session = Depends(get_session)) -> UserOutput:
    # Will create a popup to ask creds if not authenticated
    query = select(User).where(User.username == credentials.username)
    user = session.exec(query).first()
    if user and user.verify_password(credentials.password):
        return UserOutput.from_orm(user)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Username or password is incorrect"
        )

@router.post("/")
def add_car(car_input: CarInput, 
            session: Session = Depends(get_session), 
            user: User = Depends(get_current_user)) -> Car:
    # If not authenticated will require authentication
    pass
```

### OAuth2

Can be used to authenticate via other mechanisms (e.g. Facebook, Google, etc).

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, HTTPException, APIRouter
from sqlmodel import Session, select
from starlette import status
from db import get_session
from schemas import UserOutput, User

URL_PREFIX = '/AUTH'
router = APIRouter(prefix=URL_PREFIX)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{URL_PREFIX}/token")

def get_current_user(token: str = Depends(oauth2_scheme), 
                      session: Session = Depends(get_session)) -> UserOutput:
    query = select(User).where(User.username == token)
    user = session.exec(query).first()
    if user:
        return UserOutput.from_orm(user)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Wrong credentials",
            headers={"WWW_Authenticate": "Bearer"}
        )

# JWT EXAMPLE creates token data
def generate_token_data(username):
    ldapgroups = get_ldap_user_groups(username)
    payload = {
        "sub": username,
        "scopes": get_roles(username, ldapgroups),
        "alm_groups": get_alm_groups(username, ldap_groups=ldapgroups)
    }
    return payload

# JWT EXAMPLE creates token
def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=CONST_JWT_MAX_MIN)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, CONST_JWT_KEY, algorithm=CONST_JWT_ALG)
    return encoded_jwt

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), 
                session: Session = Depends(get_session)):
    # Will try to create login form if not authenticated
    query = select(User).where(User.username == form_data.username)
    user = session.exec(query).first()
    if user and user.verify_password(form_data.password):
        # JWT example
        access_token = create_access_token(data=auth.generate_token_data(form_data.username))
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
```

### LDAP Authentication

```python
import ldap3
import re

def ldap_authenticate(username, password):
    server = ldap3.Server(CONST_LDAP_SERVER, use_ssl=True, port=int(CONST_LDAP_PORT))
    ldap3.Connection(server, 'uid=' + username + ',' + CONST_LDAP_BASEDN, password, auto_bind=True)

def ldap_list_user_groups(username):
    server = ldap3.Server(CONST_LDAP_SERVER, use_ssl=True, port=int(CONST_LDAP_PORT))
    connection = ldap3.Connection(server, 'uid=' + config['ldap_username'] + ',' + CONST_LDAP_BASEDN, 
                                  config['ldap_password'], auto_bind=True)
    connection.search('OU=People,DC=fmr,DC=com', '(uid=' + username + ')', attributes=["*"])
    groups = []
    for group in connection.entries[0]['memberOf']:
        group_split = re.search('CN=(.*?),', group)
        groups.append(group_split.group(1))
    return groups

def get_roles(username, ldap_groups=None):
    if ldap_groups is None:
        ldap_groups = ldap_list_user_groups(username)
    roles = []
    for group in ldap_groups:
        if group in CONST_ADMINISTRATOR_GROUPS and 'administrators' not in roles:
            roles.append("administrators")
        for alm_role in CONST_ALM_ROLES:
            if alm_role in group.lower() and alm_role not in roles:
                roles.append(alm_role)
    return roles

def get_alm_groups(username, ldap_groups=None):
    if ldap_groups is None:
        ldap_groups = ldap_list_user_groups(username)
    groups = []
    for group in ldap_groups:
        if 'azgalm' in group.lower() and group not in groups:
            groups.append(group)
    return groups
```

### Restrict Access to an Endpoint by Role

```python
from pydantic import BaseModel
from typing import Optional, List

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []
    alm_groups: List[str] = []

def describe_current_user(security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(token, CONST_JWT_KEY, algorithms=[CONST_JWT_ALG])
        username: str = payload.get("sub")
        alm_groups: str = payload.get("alm_groups")
        if username is None:
            credentials_exception.detail = "No username found"
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(scopes=token_scopes, username=username)
    except (PyJWTError, ValidationError) as err:
        credentials_exception.detail = f"Validation error. {str(err)}"
        raise credentials_exception from err
    
    authorized = False
    if 'administrators' in token_data.scopes:
        authorized = True
    else:
        if len(security_scopes.scopes) > 0:
            for scope in security_scopes.scopes:
                if scope in token_data.scopes:
                    authorized = True
        else:
            authorized = True
    
    if authorized is False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to perform that function",
            headers={"WWW-Authenticate": authenticate_value},
        )
    
    return TokenData(
        username=username,
        scopes=token_data.scopes,
        alm_groups=alm_groups
    )

# User is required to be part of the opsengr role
@app.post("/debug-level", tags=["Logs", "Cloudops"])
def set_debug(debug_level: str, 
              current_user: TokenData = Security(describe_current_user, scopes=["opsengr"])):
    return {"Log Level": log_sys.loglevel}

# Get user information but not required to be part of a specific role
@app.get("/authorize-deploy", tags=["Authorization"], summary="Verify that request to deploy can be allowed")
async def is_auth(app_id: str, aws_acnt: str, env: str, 
                  current_user: auth.TokenData = Security(auth.describe_current_user)):
    resp = await user_auth.is_authorized(application_id=app_id, user=current_user, 
                                         aws_account=aws_acnt, environment=env)
    return json.dumps(dataclasses.asdict(resp))
```

## Custom FastAPI Exceptions

```python
from fastapi import FastAPI, Response, Depends, Header, Security, Request
from starlette.responses import JSONResponse

class AuthorizationException(Exception):
    pass

@app.exception_handler(AuthorizationException)
async def auth_exception_handler(request: Request, err: AuthorizationException):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"message": f"You are not authorized. {str(err)}"}
    )

# Example calling it in your code
raise AuthorizationException("message")
```

## When to use Async

**Documentation:** https://fastapi.tiangolo.com/async/

### Use Async when:
- You are using third party libraries that tell you to call them with `await`
- Your endpoint/method doesn't have to communicate with anything else and wait for it to respond

### Use regular `def` when:
- You are using a third party library that communicates with something (a database, an API, the file system, etc.) and doesn't have support for using `await` (this is currently the case for most database libraries)
- You aren't sure

### Asynchronous Code in Detail

Asynchronous code just means that the language has a way to tell the computer/program that at some point in the code, it will have to wait for something else to finish somewhere else.

During that time, the computer can go and do some other work, while that something else finishes. Then the computer/program will come back every time it has a chance because it's waiting again, or whenever it finished all the work it had at that point.

That "wait for something else" normally refers to I/O operations that are relatively "slow" (compared to the speed of the processor and the RAM memory), like waiting for:
- The data from the client to be sent through the network
- The data sent by your program to be received by the client through the network
- The contents of a file in the disk to be read by the system and given to your program
- The contents your program gave to the system to be written to disk
- A remote API operation
- A database operation to finish
- A database query to return the results
- etc.

As the execution time is consumed mostly by waiting for I/O operations, they call them "I/O bound" operations.

### Concurrency vs Parallelism

**Concurrency** and **parallelism** both relate to "different things happening more or less at the same time", but the details between concurrency and parallelism are quite different.

#### Concurrent Burgers

You go with your crush to get fast food. You stand in line while the cashier takes the orders from the people in front of you.

Then it's your turn, you place your order of 2 very fancy burgers for your crush and you.

The cashier says something to the cook in the kitchen so they know they have to prepare your burgers (even though they are currently preparing the ones for the previous clients).

You pay.

The cashier gives you the number of your turn.

While you are waiting, you go with your crush and pick a table, you sit and talk with your crush for a long time (as your burgers are very fancy and take some time to prepare).

As you are sitting at the table with your crush, while you wait for the burgers, you can spend that time admiring how awesome, cute and smart your crush is.

While waiting and talking to your crush, from time to time, you check the number displayed on the counter to see if it's your turn already.

Then at some point, it finally is your turn. You go to the counter, get your burgers and come back to the table.

You and your crush eat the burgers and have a nice time.

**Imagine you are the computer/program in that story.** While you are at the line, you are just idle, waiting for your turn, not doing anything very "productive". But the line is fast because the cashier is only taking the orders (not preparing them), so that's fine.

Then, when it's your turn, you do actual "productive" work, you process the menu, decide what you want, get your crush's choice, pay, check that you give the correct bill or card, check that you are charged correctly, check that the order has the correct items, etc.

But then, even though you still don't have your burgers, your work with the cashier is "on pause", because you have to wait for your burgers to be ready.

But as you go away from the counter and sit at the table with a number for your turn, you can switch your attention to your crush, and "work" on that. Then you are again doing something very "productive" as is flirting with your crush.

#### Parallel Burgers

Now let's imagine these aren't "Concurrent Burgers", but "Parallel Burgers".

You go with your crush to get parallel fast food.

You stand in line while several (let's say 8) cashiers that at the same time are cooks take the orders from the people in front of you.

Everyone before you is waiting for their burgers to be ready before leaving the counter because each of the 8 cashiers goes and prepares the burger right away before getting the next order.

Then it's finally your turn, you place your order of 2 very fancy burgers for your crush and you.

You pay.

The cashier goes to the kitchen.

You wait, standing in front of the counter, so that no one else takes your burgers before you do, as there are no numbers for turns.

As you and your crush are busy not letting anyone get in front of you and take your burgers whenever they arrive, you cannot pay attention to your crush. This is "synchronous" work, you are "synchronized" with the cashier/cook.

Then your cashier/cook finally comes back with your burgers, after a long time waiting there in front of the counter.

You take your burgers and go to the table with your crush.

You just eat them, and you are done.

There was not much talk or flirting as most of the time was spent waiting in front of the counter.

#### Burger Conclusion

In this scenario of "fast food burgers with your crush", as there is a lot of waiting, it makes a lot more sense to have a concurrent system.

This is the case for most of the web applications.

Many, many users, but your server is waiting for their not-so-good connection to send their requests.

And then waiting again for the responses to come back.

This "waiting" is measured in microseconds, but still, summing it all, it's a lot of waiting in the end.

That's why it makes a lot of sense to use asynchronous code for web APIs.

This kind of asynchronicity is what made NodeJS popular (even though NodeJS is not parallel) and that's the strength of Go as a programming language.

And that's the same level of performance you get with FastAPI.

And as you can have parallelism and asynchronicity at the same time, you get higher performance than most of the tested NodeJS frameworks and on par with Go, which is a compiled language closer to C (all thanks to Starlette).

### Is concurrency better than parallelism?

Nope! That's not the moral of the story.

Concurrency is different than parallelism. And it is better on specific scenarios that involve a lot of waiting. Because of that, it generally is a lot better than parallelism for web application development. But not for everything.

So, to balance that out, imagine the following short story:

**You have to clean a big, dirty house.**

There's no waiting anywhere, just a lot of work to be done, on multiple places of the house.

You could have turns as in the burgers example, first the living room, then the kitchen, but as you are not waiting for anything, just cleaning and cleaning, the turns wouldn't affect anything.

It would take the same amount of time to finish with or without turns (concurrency) and you would have done the same amount of work.

But in this case, if you could bring the 8 ex-cashier/cooks/now-cleaners, and each one of them (plus you) could take a zone of the house to clean it, you could do all the work in parallel, with the extra help, and finish much sooner.

In this scenario, each one of the cleaners (including you) would be a processor, doing their part of the job.

And as most of the execution time is taken by actual work (instead of waiting), and the work in a computer is done by a CPU, they call these problems "CPU bound".

Common examples of CPU bound operations are things that require complex math processing:
- **Audio or image processing**
- **Computer vision:** an image is composed of millions of pixels, each pixel has 3 values/colors, processing that normally requires computing something on those pixels, all at the same time
- **Machine Learning:** it normally requires lots of "matrix" and "vector" multiplications. Think of a huge spreadsheet with numbers and multiplying all of them together at the same time
- **Deep Learning:** this is a sub-field of Machine Learning, so, the same applies

### Concurrency + Parallelism: Web + Machine Learning

With FastAPI you can take the advantage of concurrency that is very common for web development (the same main attractive of NodeJS).

But you can also exploit the benefits of parallelism and multiprocessing (having multiple processes running in parallel) for CPU bound workloads like those in Machine Learning systems.

That, plus the simple fact that Python is the main language for Data Science, Machine Learning and especially Deep Learning, make FastAPI a very good match for Data Science/Machine Learning web APIs and applications (among many others).

### async and await

Modern versions of Python have a very intuitive way to define asynchronous code. This makes it look just like normal "sequential" code and do the "awaiting" for you at the right moments.

When there is an operation that will require waiting before giving the results and has support for these new Python features, you can code it like:

```python
burgers = await get_burgers(2)
```

The key here is the `await`. It tells Python that it has to wait for `get_burgers(2)` to finish doing its thing before storing the results in `burgers`. With that, Python will know that it can go and do something else in the meanwhile (like receiving another request).

For `await` to work, it has to be inside a function that supports this asynchronicity. To do that, you just declare it with `async def`:

```python
async def get_burgers(number: int):
    # Do some asynchronous stuff to create the burgers
    return burgers
```

...instead of `def`:

```python
# This is not asynchronous
def get_sequential_burgers(number: int):
    # Do some sequential stuff to create the burgers
    return burgers
```

With `async def`, Python knows that, inside that function, it has to be aware of `await` expressions, and that it can "pause" the execution of that function and go do something else before coming back.

When you want to call an `async def` function, you have to "await" it. So, this won't work:

```python
# This won't work, because get_burgers was defined with: async def
burgers = get_burgers(2)
```

So, if you are using a library that tells you that you can call it with `await`, you need to create the path operation functions that uses it with `async def`:

```python
@app.get('/burgers')
async def read_burgers():
    burgers = await get_burgers(2)
    return burgers
```

### More Technical Details

You might have noticed that `await` can only be used inside of functions defined with `async def`.

But at the same time, functions defined with `async def` have to be "awaited". So, functions with `async def` can only be called inside of functions defined with `async def` too.

So, about the egg and the chicken, how do you call the first `async` function?

If you are working with FastAPI you don't have to worry about that, because that "first" function will be your path operation function, and FastAPI will know how to do the right thing.

But if you want to use `async`/`await` without FastAPI, you can do it as well.

### Write your own Async Code

Starlette (and FastAPI) are based on [AnyIO](https://anyio.readthedocs.io/), which makes it compatible with both Python's standard library `asyncio` and `Trio`.

In particular, you can directly use AnyIO for your advanced concurrency use cases that require more advanced patterns in your own code.

And even if you were not using FastAPI, you could also write your own async applications with AnyIO to be highly compatible and get its benefits (e.g. structured concurrency).

### Other Forms of Asynchronous Code

This style of using `async` and `await` is relatively new in the language.

But it makes working with asynchronous code a lot easier.

This same syntax (or almost identical) was also included recently in modern versions of JavaScript (in Browser and NodeJS).

But before that, handling asynchronous code was quite more complex and difficult.

In previous versions of Python, you could have used threads or Gevent. But the code is way more complex to understand, debug, and think about.

In previous versions of NodeJS / Browser JavaScript, you would have used "callbacks". Which leads to callback hell.

### Coroutines

**Coroutine** is just the very fancy term for the thing returned by an `async def` function. Python knows that it is something like a function that it can start and that it will end at some point, but that it might be paused internally too, whenever there is an `await` inside of it.

But all this functionality of using asynchronous code with `async` and `await` is many times summarized as using "coroutines". It is comparable to the main key feature of Go, the "Goroutines".

### Conclusion

Modern versions of Python have support for "asynchronous code" using something called "coroutines", with `async` and `await` syntax.

That should make more sense now.

All that is what powers FastAPI (through Starlette) and what makes it have such an impressive performance.

## Very Technical Details

If you have quite some technical knowledge (co-routines, threads, blocking, etc.) and are curious about how FastAPI handles `async def` vs normal `def`, here are the details:

### Path Operation Functions

When you declare a **path operation function** with normal `def` instead of `async def`, it is run in an external threadpool that is then awaited, instead of being called directly (as it would block the server).

If you are coming from another async framework that does not work in the way described above and you are used to defining trivial compute-only path operation functions with plain `def` for a tiny performance gain (about 100 nanoseconds), please note that in FastAPI the effect would be quite opposite. In these cases, it's better to use `async def` unless your path operation functions use code that performs blocking I/O.

Still, in both situations, chances are that FastAPI will still be faster than (or at least comparable to) your previous framework.

### Dependencies

The same applies for **dependencies**. If a dependency is a standard `def` function instead of `async def`, it is run in the external threadpool.

### Sub-dependencies

You can have multiple dependencies and **sub-dependencies** requiring each other (as parameters of the function definitions), some of them might be created with `async def` and some with normal `def`. It would still work, and the ones created with normal `def` would be called on an external thread (from the threadpool) instead of being "awaited".

### Other Utility Functions

Any other **utility function** that you call directly can be created with normal `def` or `async def` and FastAPI won't affect the way you call it.

This is in contrast to the functions that FastAPI calls for you: path operation functions and dependencies.

If your utility function is a normal function with `def`, it will be called directly (as you write it in your code), not in a threadpool. If the function is created with `async def` then you should await for that function when you call it in your code.

## Internal CRON

Let's say you need a function to run on a cron like schedule; you can use `AsyncIOScheduler`. Here is an example of it starting on startup with proper shutdown handling.

### Scheduler

```python
# app.py
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

async def cleanup():
    # Your async job you want running on a schedule
    print("cleanup running")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler(timezone="UTC")
    app.state.scheduler = scheduler
    
    # Create a scheduled job. In this case this one runs hourly, 
    # starts immediately at app startup and has a clean, safe, shutdown
    scheduler.add_job(
        cleanup,
        trigger="interval",
        hours=1,
        id="hourly_cleanup",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc)
    )
    scheduler.start()
    try:
        yield
    finally:
        # Graceful shutdown; wait for running jobs to finish
        await scheduler.shutdown(wait=True)

# Using lifespan instead of on_event("startup"). Centralized resource management.
# Better behavior in tests: TestClient triggers the lifespan automatically when used as a context manager.
# Cleaner for complex startup/shutdown ordering.
app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Using Jobstores

**Purpose:** A jobstore is the scheduler's persistence backend for job metadata (jobs, next-run times, job state). It lets jobs survive process restarts and enables centralized job management.

#### Key Jobstore Types

- **MemoryJobStore:** in-process, non-persistent (good for tests/dev only)
- **SQLAlchemyJobStore:** relational DB-backed (SQLite/Postgres/MySQL) — common production choice
- **Third-party stores:** Redis, MongoDB, etc. (via community plugins)

#### Why Use a Jobstore

- Persistence across restarts so scheduled jobs are not lost
- Centralized job registry when you want multiple processes to see the same job list (with caveats below)
- Inspect, modify, or remove jobs at runtime via the scheduler API

#### Quick Example (SQLAlchemy)

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

jobstores = {
    "default": SQLAlchemyJobStore(url="postgresql+psycopg2://user:pass@db-host/dbname")
}
scheduler = AsyncIOScheduler(jobstores=jobstores, timezone="UTC")
scheduler.start()
```

Notes:
- SQLAlchemyJobStore will create the needed table if it doesn't exist
- Use an environment-configured DB URL in production

#### Common Operations

```python
# List jobs
scheduler.get_jobs()

# Add/update job
scheduler.add_job(..., id="myjob", replace_existing=True)

# Pause/resume/remove
scheduler.pause_job(id)
scheduler.resume_job(id)
scheduler.remove_job(id)
```

#### Important Considerations & Best Practices

- **Do NOT use MemoryJobStore in production** — it loses jobs on restart
- Use a reliable RDBMS for SQLAlchemyJobStore; enable backups and monitor DB health
- **Connection management:** pool connections and avoid short-lived DB credentials in code
- **Misfire handling:** configure `job_defaults` (misfire_grace_time, coalesce, max_instances) to control missed/overlapping runs
- **Concurrency/clustering caveat:** APScheduler + a shared jobstore is not a full distributed scheduler with leader election and locking. If multiple scheduler instances point at the same jobstore they can still both run the same job unless you implement an external leader/lock or ensure only one active scheduler runs jobs. Common patterns:
  - Run a single active scheduler instance (recommended), use standby instances for failover
  - Implement leader election (e.g., via DB advisory locks or external orchestrator) before starting the scheduler
- **Security:** protect DB credentials, use least-privilege DB users, and secure network access

#### When to Choose Which Store

- **Short-lived dev/test:** MemoryJobStore
- **Durable single-node production:** SQLAlchemyJobStore
- **High-scale or multi-node patterns:** consider external coordination (locks/leader election) or specialized distributed job systems if you need robust multi-leader scheduling
