# FastAPI Framework Guide

A comprehensive guide to building REST APIs with FastAPI, from basics to advanced patterns.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basics](#basics)
4. [HTTP Methods (CRUD)](#http-methods-crud)
5. [Request/Response Models (Schemas)](#requestresponse-models-schemas)
6. [Path Parameters & Query Parameters](#path-parameters--query-parameters)
7. [Request Body](#request-body)
8. [Organizing Routes](#organizing-routes)
9. [File Upload/Download](#file-uploaddownload)
10. [Status Codes & Error Handling](#status-codes--error-handling)
11. [Middleware](#middleware)
12. [Authentication & Authorization](#authentication--authorization)
13. [Documentation](#documentation)

---

## Introduction

**FastAPI** is a modern, fast Python web framework for building APIs with:
- Automatic interactive documentation (Swagger UI & ReDoc)
- Built-in data validation using Pydantic
- Type hints for better code clarity and IDE support
- Excellent performance
- Async/await support

---

## Installation

```bash
pip install fastapi uvicorn
```

- **fastapi** – The framework
- **uvicorn** – ASGI server to run the application

---

## Basics

### Why
Why use APIs
* Remove logic
* Easy upgrade logic
* Same logic layer for apps, web, etc

### Hello World

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Run with: python app.py
# Or: uvicorn app:app --reload
```

### Run the Server

```bash
# Option 1: Direct execution (requires if __name__ == "__main__" block)
python app.py

# Option 2: Uvicorn command
uvicorn app:app --reload

# Visit: http://127.0.0.1:8000/
# Docs: http://127.0.0.1:8000/docs
# Alternative Docs: http://127.0.0.1:8000/redoc
```

* How to test
* * Swagger
* * Postman
* * Curl

---

## HTTP Methods (CRUD)

FastAPI decorators for different HTTP methods:

```python
from fastapi import FastAPI

app = FastAPI()

# CREATE - POST
@app.post("/items")
def create_item(name: str):
    return {"message": f"Created item: {name}"}

# READ - GET
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}

# UPDATE - PUT
@app.put("/items/{item_id}")
def update_item(item_id: int, name: str):
    return {"item_id": item_id, "new_name": name}

# DELETE - DELETE
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    return {"message": f"Deleted item {item_id}"}

# PATCH (partial update)
@app.patch("/items/{item_id}")
def patch_item(item_id: int, name: str = None):
    return {"item_id": item_id, "updated_fields": {"name": name}}
```

---

## Request/Response Models (Schemas)

Use **Pydantic models** to define data structure and validation.

### Basic Schema

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    description: str = None  # Optional field
    price: float
    in_stock: bool = True

@app.post("/items")
def create_item(item: Item):
    return {"created": item}

@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    return {"id": item_id, "name": "Widget", "price": 9.99}
```

### Schema with Validation

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    id: int = Field(..., description="Unique item ID")
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price must be positive")
    quantity: int = Field(default=1, ge=0)

@app.post("/items")
def create_item(item: Item):
    return item
```

---

## Path Parameters & Query Parameters

### Path Parameters

Parameters embedded in the URL path:

```python
@app.get("/items/{item_id}")
def get_item(item_id: int):  # Type hint converts to int
    return {"item_id": item_id}

# URL: /items/42
```

### Query Parameters

Parameters passed in the query string:

```python
@app.get("/items")
def list_items(skip: int = 0, limit: int = 10, search: str = None):
    return {"skip": skip, "limit": limit, "search": search}

# URL: /items?skip=5&limit=20&search=widget
```

### Combining Both

```python
@app.get("/users/{user_id}/items")
def get_user_items(user_id: int, skip: int = 0, limit: int = 10):
    return {"user_id": user_id, "skip": skip, "limit": limit}

# URL: /users/1/items?skip=0&limit=5
```

---

## Request Body

Send data in the request body (typically with POST/PUT):

```python
class Item(BaseModel):
    name: str
    price: float
    description: str = None

@app.post("/items")
def create_item(item: Item):
    return {"created": item, "total_cost": item.price * 1.1}
```

**cURL example:**
```bash
curl -X POST "http://127.0.0.1:8000/items" \
  -H "Content-Type: application/json" \
  -d '{"name": "Widget", "price": 9.99}'
```

---

## Organizing Routes

As your API grows, split routes into separate files using **APIRouter**.

### Structure

```
project/
├── main.py
├── routers/
│   ├── __init__.py
│   ├── items.py
│   ├── users.py
│   └── orders.py
└── models/
    └── schemas.py
```

### models/schemas.py

```python
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    name: str
    price: float

class User(BaseModel):
    id: int
    username: str
    email: str
```

### routers/items.py

```python
from fastapi import APIRouter
from models.schemas import Item

router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Not found"}}
)

@router.get("", response_model=list[Item])
def list_items():
    return []

@router.post("", response_model=Item)
def create_item(item: Item):
    return item

@router.get("/{item_id}", response_model=Item)
def get_item(item_id: int):
    return {"id": item_id, "name": "Widget", "price": 9.99}

@router.put("/{item_id}", response_model=Item)
def update_item(item_id: int, item: Item):
    return item

@router.delete("/{item_id}")
def delete_item(item_id: int):
    return {"message": f"Deleted item {item_id}"}
```

### routers/users.py

```python
from fastapi import APIRouter
from models.schemas import User

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("", response_model=list[User])
def list_users():
    return []

@router.post("", response_model=User)
def create_user(user: User):
    return user
```

### main.py

```python
from fastapi import FastAPI
from routers import items, users

app = FastAPI(
    title="My API",
    description="A simple API with organized routes",
    version="1.0.0"
)

# Include routers
app.include_router(items.router)
app.include_router(users.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to My API"}
```

---

## File Upload/Download

### File Upload

```python
from fastapi import FastAPI, File, UploadFile
from typing import List

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file"""
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}

@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    results = []
    for file in files:
        contents = await file.read()
        results.append({"filename": file.filename, "size": len(contents)})
    return results
```

### File Download

```python
from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file"""
    file_path = Path(f"files/{filename}")
    if file_path.exists():
        return FileResponse(path=file_path, filename=filename)
    return {"error": "File not found"}
```

---

## Status Codes & Error Handling

### Custom Status Codes

```python
from fastapi import FastAPI, status

app = FastAPI()

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(name: str):
    return {"name": name}

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    pass  # Returns 204 No Content
```

### Exception Handling

```python
from fastapi import HTTPException, status

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item ID must be positive"
        )
    if item_id > 1000:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )
    return {"item_id": item_id}
```

### Global Exception Handler

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

class CustomException(Exception):
    def __init__(self, message: str):
        self.message = message

@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message}
    )
```

---

## Middleware

Middleware runs on every request/response.

### Built-in Middleware

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

### Custom Middleware

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(LoggingMiddleware)
```

---

## Authentication & Authorization

### Basic Token-Based Authentication

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials

app = FastAPI()
security = HTTPBearer()

VALID_TOKEN = "my-secret-token"

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

@app.get("/protected")
async def protected_route(token: str = Depends(verify_token)):
    return {"message": "Access granted", "token": token}
```

### JWT Token Authentication

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
import jwt

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username

@app.get("/users/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}
```

---

## Documentation

FastAPI automatically generates interactive API documentation.

### Access Documentation

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:8000/openapi.json`

### Adding Descriptions & Examples

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="My Awesome API",
    description="An API for managing items",
    version="1.0.0"
)

class Item(BaseModel):
    """Schema for an item"""
    id: int = Field(..., description="Unique item identifier", example=1)
    name: str = Field(..., min_length=1, description="Item name", example="Widget")
    price: float = Field(..., gt=0, description="Item price", example=9.99)

@app.get(
    "/items/{item_id}",
    summary="Get an item",
    description="Retrieve a specific item by its ID",
    response_description="The item object"
)
def get_item(
    item_id: int = Field(..., description="The item ID", example=1)
) -> Item:
    """
    Get item details.
    
    Returns:
        Item: The requested item
    """
    return {"id": item_id, "name": "Widget", "price": 9.99}

@app.post(
    "/items",
    summary="Create an item",
    tags=["items"]
)
def create_item(item: Item):
    """Create a new item in the database"""
    return item
```

---

## Advanced Topics

### Dependency Injection

```python
from fastapi import Depends

async def get_db():
    # Setup database connection
    db = "database_connection"
    yield db
    # Cleanup

@app.get("/items")
async def get_items(db: str = Depends(get_db)):
    return {"db": db}
```

### Background Tasks

```python
from fastapi import BackgroundTasks

@app.post("/send-notification")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email)
    return {"message": "Notification sent in background"}

def send_email(email: str):
    # Simulate sending email
    print(f"Sending email to {email}")
```

### Request Validation

```python
from pydantic import BaseModel, validator

class Item(BaseModel):
    name: str
    price: float
    
    @validator("price")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
```

---

## Best Practices

1. **Organize routes** using APIRouter for large applications
2. **Use type hints** for automatic validation and documentation
3. **Separate concerns** – models, routers, database logic in different files
4. **Use dependency injection** for database connections, auth, etc.
5. **Handle errors gracefully** with meaningful error messages
6. **Document endpoints** with descriptions, examples, and tags
7. **Use async** for I/O-bound operations
8. **Test endpoints** with FastAPI's TestClient

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
```

---

## Resources

- [Official FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Starlette Documentation](https://www.starlette.io/)
