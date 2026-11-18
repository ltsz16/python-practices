# Advanced Python Topics: A Comprehensive Guide

This document provides an in-depth exploration of 16 advanced Python concepts. Each topic includes an overview, detailed explanations for beginners, and practical Python examples.

---

## 1. Multiprocessing for CPU-bound Parallelism

### Overview
**Multiprocessing** creates separate Python processes to run code in parallel on multiple CPU cores. Unlike threading (which is limited by the Global Interpreter Lock for CPU work), multiprocessing actually executes code simultaneously on different cores.

### When to Use
- **CPU-bound tasks**: Heavy computations, data processing, numerical calculations
- NOT suitable for I/O-bound tasks (use `asyncio` or `threading` instead)

### Key Concepts
- Each process has its own Python interpreter and memory space
- No Global Interpreter Lock (GIL) limitations
- Inter-process communication via queues, pipes, or shared memory
- More overhead than threading (slower to start)

### Example: CPU-Bound Work with Multiprocessing

```python
import multiprocessing
import time

def cpu_intensive_task(n):
    """Simulate heavy computation"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

def single_process_example():
    """Sequential processing"""
    start = time.time()
    results = []
    for i in range(1, 5):
        result = cpu_intensive_task(10**7)
        results.append(result)
    elapsed = time.time() - start
    print(f"Sequential: {elapsed:.2f}s")
    return results

def multiprocess_example():
    """Parallel processing"""
    start = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(cpu_intensive_task, [10**7] * 4)
    elapsed = time.time() - start
    print(f"Multiprocessing: {elapsed:.2f}s")
    return results

if __name__ == "__main__":
    print("CPU-Bound Multiprocessing Example")
    single_process_example()
    multiprocess_example()
```

### Example: Inter-Process Communication

```python
from multiprocessing import Process, Queue

def worker(queue, task_id):
    """Worker process that puts results in queue"""
    result = f"Result from task {task_id}"
    queue.put(result)

if __name__ == "__main__":
    queue = Queue()
    processes = []
    
    for i in range(3):
        p = Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    while not queue.empty():
        print(queue.get())
```

---

## 2. Descriptors and Properties

### Overview
**Descriptors** are Python objects that define how attributes are accessed, modified, or deleted. **Properties** are a specific type of descriptor that allow you to customize attribute access with methods.

### Why Use Descriptors?
- Validation: Ensure only valid values are set
- Computed properties: Calculate values on-the-fly
- Lazy loading: Load data only when needed
- Encapsulation: Hide implementation details

### Key Concepts
- Descriptor protocol: `__get__()`, `__set__()`, `__delete__()`
- `@property` decorator for read-only properties
- `@property.setter` for write access with validation
- Data descriptors vs non-data descriptors

### Example: Property with Validation

```python
class Person:
    def __init__(self, name, age):
        self._name = name
        self._age = age
    
    @property
    def age(self):
        """Read-only property"""
        return self._age
    
    @age.setter
    def age(self, value):
        """Validate age before setting"""
        if not isinstance(value, int):
            raise TypeError("Age must be an integer")
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        if len(value) < 1:
            raise ValueError("Name cannot be empty")
        self._name = value

# Usage
person = Person("Alice", 30)
print(f"{person.name} is {person.age} years old")
person.age = 31  # Valid
print(person.age)
# person.age = -5  # Raises ValueError
```

### Example: Custom Descriptor

```python
class ValidatedString:
    """A descriptor that validates string values"""
    def __init__(self, min_length=0, max_length=None):
        self.min_length = min_length
        self.max_length = max_length
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, '')
    
    def __set__(self, obj, value):
        if not isinstance(value, str):
            raise TypeError("Value must be a string")
        if len(value) < self.min_length:
            raise ValueError(f"String too short (min {self.min_length})")
        if self.max_length and len(value) > self.max_length:
            raise ValueError(f"String too long (max {self.max_length})")
        obj.__dict__[self.name] = value
    
    def __set_name__(self, owner, name):
        self.name = name

class Product:
    name = ValidatedString(min_length=1, max_length=50)
    description = ValidatedString(max_length=200)
    
    def __init__(self, name, description):
        self.name = name
        self.description = description

# Usage
product = Product("Laptop", "High-performance computer")
print(product.name)
```

---

## 3. Metaclasses (Classes That Create Classes)

### Overview
A **metaclass** is a class whose instances are classes. While `type()` is the default metaclass, you can create custom metaclasses to control how classes are created and behave.

### Why Use Metaclasses?
- Framework development (ORM, API frameworks)
- Class registration and tracking
- Enforcing class structure/inheritance
- Modifying class behavior at creation time
- Advanced metaprogramming

### Key Concepts
- Metaclasses are classes with `type` as their metaclass
- `__new__()` called when class is created
- `__init__()` called to initialize the class object
- `__call__()` called when class is instantiated

### Example: Simple Metaclass

```python
class SingletonMeta(type):
    """Metaclass that creates singleton classes"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = None
    
    def connect(self):
        self.connection = "Connected to DB"

# Usage
db1 = Database()
db2 = Database()
print(db1 is db2)  # True - same instance
print(db1.connection)
```

### Example: Class Registration Metaclass

```python
class PluginRegistry(type):
    """Metaclass that automatically registers subclasses"""
    plugins = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # Don't register base class
            mcs.plugins[name] = cls
        return cls

class Plugin(metaclass=PluginRegistry):
    def execute(self):
        raise NotImplementedError

class PDFPlugin(Plugin):
    def execute(self):
        return "Processing PDF"

class ImagePlugin(Plugin):
    def execute(self):
        return "Processing Image"

# Usage
print(PluginRegistry.plugins)
# Output: {'PDFPlugin': <class 'PDFPlugin'>, 'ImagePlugin': <class 'ImagePlugin'>}

for plugin_class in PluginRegistry.plugins.values():
    plugin = plugin_class()
    print(plugin.execute())
```

---

## 4. Slots (__slots__) for Memory Optimization

### Overview
`__slots__` is a class attribute that restricts which instance attributes can exist, saving memory by using fixed slots instead of a dynamic dictionary.

### When to Use
- Creating millions of objects (significant memory savings)
- Classes with a fixed set of attributes
- Performance-critical applications
- NOT suitable for dynamic attribute addition

### How It Works
- Without `__slots__`: Each instance has a `__dict__` (dictionary) to store attributes
- With `__slots__`: Attributes stored in fixed memory slots
- Memory savings: 40-50% reduction per instance

### Example: Memory Comparison

```python
import sys

class PersonWithoutSlots:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class PersonWithSlots:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Memory comparison
person_no_slots = PersonWithoutSlots("Alice", 30)
person_with_slots = PersonWithSlots("Alice", 30)

print(f"Without slots: {sys.getsizeof(person_no_slots.__dict__)} bytes for __dict__")
print(f"With slots: {sys.getsizeof(person_with_slots)} bytes for instance")

# Creating many objects
objects_no_slots = [PersonWithoutSlots(f"Person{i}", 20+i) for i in range(100000)]
objects_with_slots = [PersonWithSlots(f"Person{i}", 20+i) for i in range(100000)]

# Check total memory
total_no_slots = sum(sys.getsizeof(obj.__dict__) for obj in objects_no_slots)
total_with_slots = sum(sys.getsizeof(obj) for obj in objects_with_slots)

print(f"\n100,000 objects:")
print(f"Without slots total: ~{total_no_slots / 1024 / 1024:.2f} MB")
print(f"With slots total: ~{total_with_slots / 1024 / 1024:.2f} MB")
print(f"Memory saved: ~{(1 - total_with_slots/total_no_slots)*100:.1f}%")
```

### Example: Slots with Inheritance

```python
class Animal:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Dog(Animal):
    __slots__ = ['breed']  # Additional slots
    
    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

# Usage
dog = Dog("Buddy", 5, "Golden Retriever")
print(f"{dog.name} is a {dog.breed}")
# dog.color = "brown"  # Would raise AttributeError
```

---

## 5. Context Variables for Async Context Management

### Overview
**Context variables** allow you to store state that is isolated per async task. Unlike global variables, context variables maintain separate values for different async tasks running concurrently.

### Why Use Context Variables?
- Per-task logging context (request IDs, user info)
- User-local data in async applications
- Avoiding global state in concurrent code
- Thread-safe and task-safe state management

### Key Concepts
- `contextvars.ContextVar` for creating context variables
- `.get()` to retrieve values
- `.set()` to assign values
- State is isolated per async task

### Example: Context Variables in Async Tasks

```python
import asyncio
from contextvars import ContextVar

# Create context variable
user_context = ContextVar('user_id', default=None)
request_id = ContextVar('request_id', default=None)

async def log_message(message):
    """Log with context information"""
    user = user_context.get()
    req_id = request_id.get()
    print(f"[User:{user}][Request:{req_id}] {message}")

async def process_request(user_id, req_id):
    """Process request with context"""
    user_context.set(user_id)
    request_id.set(req_id)
    
    await log_message("Request started")
    await asyncio.sleep(0.1)
    await log_message("Processing data")
    await asyncio.sleep(0.1)
    await log_message("Request completed")

async def main():
    """Run multiple concurrent requests"""
    await asyncio.gather(
        process_request("user1", "req-001"),
        process_request("user2", "req-002"),
        process_request("user3", "req-003"),
    )

# asyncio.run(main())
```

### Example: Copying Context to New Tasks

```python
import asyncio
from contextvars import ContextVar, copy_context

trace_id = ContextVar('trace_id', default=None)

async def background_task():
    """Task that should inherit parent context"""
    trace = trace_id.get()
    print(f"Background task running with trace_id={trace}")

async def main_task():
    """Main task setting context"""
    trace_id.set("trace-123")
    
    # Create task with copied context
    ctx = copy_context()
    task = asyncio.create_task(ctx.run(background_task))
    await task

# asyncio.run(main_task())
```

---

## 6. Named Tuples (collections.namedtuple, typing.NamedTuple)

### Overview
**Named tuples** are lightweight, immutable data structures that combine tuple benefits (memory efficiency, immutability) with named field access (like dictionaries).

### Why Use Named Tuples?
- Lightweight alternative to classes for simple data
- Immutable by design (prevents accidental modifications)
- Memory-efficient
- Better than plain tuples for code readability
- Can be used as dictionary keys (unlike lists or regular class instances)

### Comparison
- `collections.namedtuple`: Simple, lightweight, classic
- `typing.NamedTuple`: Type hints, cleaner syntax, Python 3.6+

### Example: collections.namedtuple

```python
from collections import namedtuple

# Define a Point namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Create instances
p1 = Point(10, 20)
p2 = Point(x=15, y=25)

# Access fields
print(f"Point coordinates: ({p1.x}, {p1.y})")
print(f"Unpacking: {p1[0]}, {p1[1]}")

# Named tuples are immutable
# p1.x = 30  # AttributeError

# Can be used as dict keys
locations = {p1: "Office", p2: "Home"}
print(locations)

# Useful methods
print(f"Fields: {Point._fields}")
print(f"As dict: {p1._asdict()}")
print(f"Replace: {p1._replace(x=100)}")
```

### Example: typing.NamedTuple with Type Hints

```python
from typing import NamedTuple

class Employee(NamedTuple):
    name: str
    employee_id: int
    department: str
    salary: float

# Create instances
emp1 = Employee("Alice", 101, "Engineering", 95000)
emp2 = Employee(name="Bob", employee_id=102, department="Sales", salary=75000)

# Access fields
print(f"{emp1.name} works in {emp1.department}")

# Type hints provide IDE support and type checking
employees = [emp1, emp2]
for emp in employees:
    print(f"{emp.name}: ${emp.salary}")
```

### Example: Named Tuples in Functions

```python
from typing import NamedTuple

class QueryResult(NamedTuple):
    success: bool
    data: list
    error: str | None = None

def query_database(sql: str) -> QueryResult:
    """Simulate database query"""
    if "SELECT" in sql:
        return QueryResult(success=True, data=[1, 2, 3])
    else:
        return QueryResult(success=False, data=[], error="Invalid query")

# Usage
result = query_database("SELECT * FROM users")
if result.success:
    print(f"Got data: {result.data}")
else:
    print(f"Error: {result.error}")
```

---

## 7. Pickle/Serialization for Object Persistence

### Overview
**Pickle** is Python's built-in module for serializing (converting to bytes) and deserializing (converting back to objects) Python objects. This allows saving objects to disk or sending them over networks.

### When to Use
- Saving Python objects to disk
- Caching complex objects
- Inter-process communication
- Note: DON'T unpickle untrusted data (security risk)

### Key Concepts
- Serialization: Object → bytes
- Deserialization: bytes → Object
- Works with most Python objects
- Alternative formats: JSON (text), MessagePack (binary), Protocol Buffers

### Example: Basic Pickling

```python
import pickle

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

# Create and pickle object
person = Person("Alice", 30)
print(f"Original: {person}")

# Serialize to file
with open('person.pkl', 'wb') as f:
    pickle.dump(person, f)

# Deserialize from file
with open('person.pkl', 'rb') as f:
    loaded_person = pickle.load(f)

print(f"Loaded: {loaded_person}")
print(f"Name: {loaded_person.name}, Age: {loaded_person.age}")

# Also works with bytes
pickled_bytes = pickle.dumps(person)
unpickled = pickle.loads(pickled_bytes)
print(f"From bytes: {unpickled}")
```

### Example: Pickling Complex Objects

```python
import pickle
from datetime import datetime

class Order:
    def __init__(self, order_id, items, timestamp=None):
        self.order_id = order_id
        self.items = items
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self):
        return f"Order({self.order_id}, {len(self.items)} items)"

# Create complex object
orders = [
    Order(1, ["Laptop", "Mouse"]),
    Order(2, ["Phone", "Charger", "Case"]),
    Order(3, ["Monitor"]),
]

# Pickle the list
pickled = pickle.dumps(orders)
print(f"Pickled size: {len(pickled)} bytes")

# Unpickle
restored_orders = pickle.loads(pickled)
for order in restored_orders:
    print(f"Order {order.order_id}: {order.items} (saved on {order.timestamp})")
```

### Example: Custom Pickling with __getstate__ and __setstate__

```python
import pickle

class ConfigFile:
    def __init__(self, filename, settings):
        self.filename = filename
        self.settings = settings
        self.file_handle = open(filename, 'r')  # Not pickleable
    
    def __getstate__(self):
        """Return state to pickle (exclude file handle)"""
        state = self.__dict__.copy()
        del state['file_handle']  # Remove unpickleable attribute
        return state
    
    def __setstate__(self, state):
        """Restore state from pickle"""
        self.__dict__.update(state)
        self.file_handle = open(self.filename, 'r')

# Usage
config = ConfigFile('config.txt', {'debug': True, 'port': 8000})
pickled = pickle.dumps(config)
restored = pickle.loads(pickled)
print(f"Restored config: {restored.settings}")
```

---

## 8. Networking (Sockets, Client-Server Communication)

### Overview
**Sockets** are endpoints for network communication. The `socket` module provides low-level networking interface for TCP/IP communication between clients and servers.

### Key Concepts
- **Server**: Listens for incoming connections
- **Client**: Initiates connection to server
- **Socket types**: TCP (SOCK_STREAM) for reliable, UDP (SOCK_DGRAM) for fast
- **Ports**: Numbered endpoints (0-65535)

### Example: Simple TCP Server

```python
import socket
import threading

def tcp_server():
    """Simple TCP server"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 5000))
    server.listen(1)
    print("Server listening on port 5000...")
    
    try:
        client, addr = server.accept()
        print(f"Connection from {addr}")
        
        # Receive data
        data = client.recv(1024).decode()
        print(f"Received: {data}")
        
        # Send response
        client.send(f"Echo: {data}".encode())
        client.close()
    finally:
        server.close()

def tcp_client():
    """Simple TCP client"""
    import time
    time.sleep(1)  # Wait for server to start
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))
    
    # Send data
    message = "Hello, Server!"
    client.send(message.encode())
    
    # Receive response
    response = client.recv(1024).decode()
    print(f"Server replied: {response}")
    
    client.close()

# Run server and client
# server_thread = threading.Thread(target=tcp_server)
# client_thread = threading.Thread(target=tcp_client)
# server_thread.start()
# client_thread.start()
```

### Example: UDP Communication (Fire-and-Forget)

```python
import socket
import threading
import time

def udp_server():
    """UDP server (simpler than TCP)"""
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(('localhost', 5001))
    print("UDP Server listening on port 5001...")
    
    for _ in range(3):
        data, addr = server.recvfrom(1024)
        print(f"From {addr}: {data.decode()}")
    
    server.close()

def udp_client():
    """UDP client"""
    time.sleep(0.5)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    for i in range(3):
        message = f"Message {i}"
        client.sendto(message.encode(), ('localhost', 5001))
        time.sleep(0.1)
    
    client.close()

# Run UDP server and client
# server_thread = threading.Thread(target=udp_server)
# client_thread = threading.Thread(target=udp_client)
# server_thread.start()
# client_thread.start()
```

---

## 9. Database ORMs (SQLAlchemy)

### Overview
An **Object-Relational Mapping (ORM)** library maps database tables to Python classes, allowing you to work with databases using Python objects instead of raw SQL.

### Benefits of ORMs
- No SQL needed: Use Python objects
- Type-safe: IDE support and type hints
- Automatic validation: Built-in constraints
- Database-agnostic: Switch databases easily
- Prevents SQL injection: Automatic escaping

### Key SQLAlchemy Concepts
- **Model**: Python class representing database table
- **Session**: Manages database operations
- **Relationships**: Link between models
- **Query**: Retrieve data from database

### Example: Basic SQLAlchemy Model

```python
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    
    def __repr__(self):
        return f"Product(name={self.name!r}, price={self.price}, stock={self.stock})"

# Create in-memory SQLite database
engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)

# Create products
with Session(engine) as session:
    # Add products
    products = [
        Product(name="Laptop", price=999.99, stock=5),
        Product(name="Mouse", price=29.99, stock=50),
        Product(name="Keyboard", price=79.99, stock=25),
    ]
    session.add_all(products)
    session.commit()
    
    # Query products
    all_products = session.query(Product).all()
    for product in all_products:
        print(product)
    
    # Filter query
    expensive = session.query(Product).filter(Product.price > 100).all()
    print(f"\nExpensive products: {expensive}")
```

### Example: Relationships Between Models

```python
from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, Session, relationship

Base = declarative_base()

class Author(Base):
    __tablename__ = "authors"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    books = relationship("Book", back_populates="author")

class Book(Base):
    __tablename__ = "books"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    author_id = Column(Integer, ForeignKey("authors.id"))
    author = relationship("Author", back_populates="books")

# Usage
engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)

with Session(engine) as session:
    # Create author with books
    author = Author(name="George R.R. Martin")
    author.books = [
        Book(title="A Game of Thrones"),
        Book(title="A Clash of Kings"),
    ]
    session.add(author)
    session.commit()
    
    # Query with relationship
    author = session.query(Author).first()
    for book in author.books:
        print(f"{author.name} wrote '{book.title}'")
```

---

## 10. Web Frameworks Beyond Flask/Django (FastAPI, Starlette)

### Overview
Modern async web frameworks offer high performance and clean code. **FastAPI** combines speed with automatic API documentation, while **Starlette** provides lightweight async components.

### Advantages
- **FastAPI**: Built on Starlette, automatic docs (Swagger/OpenAPI), type hints
- **Starlette**: Minimal, flexible, async-first
- **Async**: Better performance for I/O-heavy operations

### Example: FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Simple API")

# Define data model
class Item(BaseModel):
    name: str
    price: float
    is_available: bool = True

# In-memory database
items_db = [
    Item(name="Laptop", price=999.99),
    Item(name="Mouse", price=29.99),
]

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to Simple API"}

@app.get("/items", response_model=List[Item])
def get_items():
    """Get all items"""
    return items_db

@app.get("/items/{item_id}")
def get_item(item_id: int):
    """Get specific item by ID"""
    if item_id < 0 or item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

@app.post("/items", response_model=Item)
def create_item(item: Item):
    """Create new item"""
    items_db.append(item)
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    """Update existing item"""
    if item_id < 0 or item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_id] = item
    return item

# Run with: uvicorn script_name:app --reload
# Automatic docs at: http://localhost:8000/docs
```

### Example: Starlette Application

```python
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import asyncio

async def homepage(request):
    """Homepage handler"""
    return JSONResponse({"message": "Hello, Starlette!"})

async def slow_operation(request):
    """Async operation - doesn't block"""
    user_id = request.path_params['user_id']
    await asyncio.sleep(1)  # Simulate I/O
    return JSONResponse({"user_id": user_id, "data": "loaded"})

routes = [
    Route('/', homepage),
    Route('/users/{user_id}', slow_operation),
]

app = Starlette(routes=routes)

# Run with: uvicorn script_name:app
```

---

## 11. Machine Learning Frameworks (scikit-learn, TensorFlow, PyTorch, JAX)

### Overview
Machine learning frameworks provide tools for building predictive models. Each has different strengths:

- **scikit-learn**: Simple, classical ML algorithms
- **TensorFlow**: Production-ready deep learning
- **PyTorch**: Research-friendly deep learning
- **JAX**: Functional computing with automatic differentiation

### Example: scikit-learn for Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### Example: PyTorch Neural Network

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# Define simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)      # Input layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)      # Hidden layer
        self.fc3 = nn.Linear(8, 3)       # Output layer (3 classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model, optimizer, loss
model = SimpleNN()
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop (simplified)
def train_step(x, y):
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Example usage
X_sample = torch.randn(32, 4)  # 32 samples, 4 features
y_sample = torch.randint(0, 3, (32,))  # 32 labels (3 classes)
loss = train_step(X_sample, y_sample)
print(f"Loss: {loss:.4f}")
```

---

## 12. Data Visualization (Matplotlib, Plotly, Seaborn)

### Overview
Visualization libraries help explore and communicate data insights. Each has different strengths:

- **Matplotlib**: Foundational, highly customizable, static plots
- **Seaborn**: Statistical visualization, beautiful defaults
- **Plotly**: Interactive plots, web-ready

### Example: Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Line plot
ax1.plot(x, y1, label='sin(x)', linewidth=2)
ax1.plot(x, y2, label='cos(x)', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True)

# Plot 2: Scatter plot
y_noise = np.sin(x) + np.random.normal(0, 0.1, len(x))
ax2.scatter(x, y_noise, alpha=0.5, s=50)
ax2.set_xlabel('x')
ax2.set_ylabel('y + noise')
ax2.set_title('Noisy Sine Wave')

plt.tight_layout()
plt.show()
```

### Example: Seaborn

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Generate sample data
data = {
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'] * 10,
    'Value': np.random.randint(10, 100, 60),
    'Group': ['Group1', 'Group2'] * 30,
}
df = pd.DataFrame(data)

# Create beautiful statistical plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot
sns.boxplot(data=df, x='Category', y='Value', ax=axes[0, 0])
axes[0, 0].set_title('Distribution by Category')

# Violin plot
sns.violinplot(data=df, x='Category', y='Value', hue='Group', ax=axes[0, 1])
axes[0, 1].set_title('Violin Plot with Groups')

# Count plot
sns.countplot(data=df, x='Category', ax=axes[1, 0])
axes[1, 0].set_title('Count by Category')

# Heatmap (correlation matrix)
sample_data = np.random.randn(10, 5)
sns.heatmap(np.corrcoef(sample_data.T), annot=True, ax=axes[1, 1], cmap='coolwarm')
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()
```

---

## 13. Package Development and Distribution

### Overview
Publishing Python packages allows sharing code with the community. Modern tools make this straightforward.

### Tools
- **setuptools**: Traditional packaging
- **poetry**: Modern dependency management and packaging
- **pyproject.toml**: Unified project configuration (PEP 517/518)

### Example: Poetry Project Structure

```
my_package/
├── pyproject.toml              # Project configuration
├── README.md
├── LICENSE
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
└── tests/
    └── test_module1.py
```

### Example: pyproject.toml (Poetry)

```toml
[tool.poetry]
name = "my-awesome-package"
version = "0.1.0"
description = "A brief description"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
license = "MIT"

[tool.poetry.dependencies]
python = "^3.9"
numpy = "^1.21.0"
pandas = "^1.3.0"

[tool.poetry.dev-dependencies]
pytest = "^6.2"
black = "^21.0"
mypy = "^0.910"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### Example: setup.py (Traditional)

```python
from setuptools import setup, find_packages

setup(
    name="my-awesome-package",
    version="0.1.0",
    description="A brief description",
    author="Your Name",
    author_email="you@example.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
```

### Publishing to PyPI

```bash
# With Poetry
poetry build                    # Build package
poetry publish                  # Publish to PyPI

# With setuptools
python setup.py sdist bdist_wheel   # Create distribution
twine upload dist/*                 # Upload to PyPI
```

---

## 14. Garbage Collection and Memory Management

### Overview
Python uses **automatic memory management** with garbage collection. Understanding this helps write efficient code and debug memory issues.

### Key Concepts
- **Reference counting**: Objects freed when reference count reaches zero
- **Garbage collection**: Detects and frees circular references
- **Memory leaks**: Accidental reference retention
- `gc` module: Controls garbage collection behavior

### Example: Reference Counting

```python
import sys

class MyObject:
    def __init__(self, name):
        self.name = name

obj = MyObject("example")
print(f"Reference count: {sys.getrefcount(obj)}")  # 2 (obj + function arg)

# Create more references
ref1 = obj
ref2 = obj
print(f"Reference count: {sys.getrefcount(obj)}")  # 4

# Delete references
del ref1
del ref2
print(f"Reference count: {sys.getrefcount(obj)}")  # 2

# When obj goes out of scope, reference count = 0, memory freed
```

### Example: Circular References and Garbage Collection

```python
import gc
import weakref

class Node:
    def __init__(self, name):
        self.name = name
        self.next = None

# Create circular reference
node1 = Node("A")
node2 = Node("B")
node1.next = node2
node2.next = node1  # Circular!

# Even deleting variables won't free memory (circular reference)
del node1
del node2

# Garbage collector detects and frees circular references
collected = gc.collect()
print(f"Garbage collector freed {collected} objects")
```

### Example: Memory Monitoring

```python
import tracemalloc
import numpy as np

tracemalloc.start()

# Allocate memory
data = [list(range(1000)) for _ in range(1000)]
snapshot = tracemalloc.take_snapshot()

# Show top memory consumers
top_stats = snapshot.statistics('lineno')
print("[ Top 3 ]")
for stat in top_stats[:3]:
    print(stat)
```

---

## 15. Bytecode and CPython Internals

### Overview
Python code is compiled to **bytecode** (intermediate representation) executed by the CPython virtual machine. Understanding this helps optimize performance.

### Key Concepts
- **Compilation**: Source code → bytecode (.pyc files)
- **Bytecode**: Low-level instructions (LOAD_CONST, STORE_FAST, etc.)
- **Virtual machine**: Executes bytecode
- **.pyc cache**: Stores compiled bytecode for faster loading

### Example: Inspecting Bytecode

```python
import dis

def example_function(x):
    if x > 0:
        return x * 2
    else:
        return -x

# Disassemble function
print("Bytecode for example_function:")
dis.dis(example_function)

# Output shows bytecode instructions:
# LOAD_FAST, LOAD_CONST, COMPARE_OP, POP_JUMP_IF_FALSE, etc.
```

### Example: Bytecode Optimization

```python
import timeit

# Slow: attribute lookup in loop
def slow_loop():
    result = 0
    for i in range(1000):
        result += i

# Fast: local reference
def fast_loop():
    result = 0
    local_range = range
    for i in local_range(1000):
        result += i

slow_time = timeit.timeit(slow_loop, number=10000)
fast_time = timeit.timeit(fast_loop, number=10000)

print(f"Slow: {slow_time:.4f}s")
print(f"Fast: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

### Example: Viewing .pyc Files

```python
import py_compile
import importlib.util

# Compile Python file to bytecode
py_compile.compile('example.py', cfile='example.cpython-39.pyc')

# Load and execute bytecode
spec = importlib.util.spec_from_file_location("example", 'example.cpython-39.pyc')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

---

## 16. C Extensions and ctypes for Performance-Critical Code

### Overview
For performance-critical code, you can call C/C++ libraries from Python using:
- **ctypes**: Call C functions from Python (easiest)
- **C extensions**: Write Python modules in C (most control)
- **Cython**: Write C-like Python code (hybrid)

### When to Use
- Computationally intensive operations
- Existing C libraries to use
- Performance critical sections (profiling first!)

### Example: ctypes - Calling C Library

```python
import ctypes
import sys

# Load C math library
if sys.platform == 'win32':
    libm = ctypes.CDLL('msvcrt')
else:
    libm = ctypes.CDLL('libm.so.6')

# Define function signatures
libm.sin.argtypes = [ctypes.c_double]
libm.sin.restype = ctypes.c_double

libm.sqrt.argtypes = [ctypes.c_double]
libm.sqrt.restype = ctypes.c_double

# Call C functions
import math
angle = 0.5
print(f"C sqrt(16) = {libm.sqrt(16.0)}")
print(f"Python sqrt(16) = {math.sqrt(16.0)}")
print(f"C sin(0.5) = {libm.sin(angle)}")
print(f"Python sin(0.5) = {math.sin(angle)}")
```

### Example: ctypes - Calling Custom C Function

```c
// In file: mylib.c
#include <math.h>

double add(double a, double b) {
    return a + b;
}

double multiply(double a, double b) {
    return a * b;
}

// Compile: gcc -shared -fPIC -o mylib.so mylib.c (Linux)
//          gcc -shared -o mylib.dll mylib.c (Windows)
```

```python
import ctypes
import os

# Load shared library
if os.path.exists('mylib.so'):
    mylib = ctypes.CDLL('./mylib.so')
else:
    mylib = ctypes.CDLL('./mylib.dll')

# Define function signatures
mylib.add.argtypes = [ctypes.c_double, ctypes.c_double]
mylib.add.restype = ctypes.c_double

mylib.multiply.argtypes = [ctypes.c_double, ctypes.c_double]
mylib.multiply.restype = ctypes.c_double

# Call functions
result_add = mylib.add(3.5, 2.5)
result_mult = mylib.multiply(4.0, 5.0)

print(f"3.5 + 2.5 = {result_add}")
print(f"4.0 * 5.0 = {result_mult}")
```

### Example: Performance Comparison

```python
import ctypes
import timeit

# Pure Python implementation
def python_factorial(n):
    if n <= 1:
        return 1
    return n * python_factorial(n - 1)

# Using math.factorial (C implementation)
import math

# Benchmark
n = 20
python_time = timeit.timeit(lambda: python_factorial(n), number=10000)
c_time = timeit.timeit(lambda: math.factorial(n), number=10000)

print(f"Python implementation: {python_time:.4f}s")
print(f"C implementation: {c_time:.4f}s")
print(f"Speedup: {python_time/c_time:.0f}x faster")
```

---

## 17. Dunder Methods (Magic Methods) — Customizing Object Behavior

### Overview
**Dunder methods** (double underscore methods, also called "magic methods") are special methods that define how objects behave with Python's built-in functions and operators. They're called automatically by Python in specific contexts, making your custom objects feel like built-in types.

### Why Use Dunders?
- Make custom objects work with Python built-ins (`len()`, `str()`, iteration, etc.)
- Enable operator overloading (`+`, `-`, `==`, `<`, etc.)
- Create intuitive, Pythonic APIs
- Enable context managers and iterators
- Make objects callable like functions

---

### `__init__()` — Constructor/Initializer

```python
class BankAccount:
    """Account initialization with dunder __init__"""
    
    def __init__(self, owner, balance=0):
        """Called when creating new instance"""
        self.owner = owner
        self.balance = balance
    
    def __repr__(self):
        return f"BankAccount(owner='{self.owner}', balance=${self.balance})"

# Usage: __init__ is called automatically
account = BankAccount("Alice", 1000)
print(account)  # BankAccount(owner='Alice', balance=$1000)

# Without __init__, you'd have to manually set attributes
class BadAccount:
    pass

bad_account = BadAccount()
# bad_account.owner = "Bob"  # Manual assignment - error-prone
```

---

### `__str__()` — String Representation for Users (Readable)

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        """Called by str() and print() - user-friendly"""
        return f"{self.name} is {self.age} years old"

person = Person("Alice", 30)
print(person)  # Alice is 30 years old (uses __str__)
print(str(person))  # Alice is 30 years old

# Without __str__, you'd see:
# <__main__.Person object at 0x7f8b8c0d5f40>
```

---

### `__repr__()` — Developer Representation (Unambiguous)

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        """Called by repr() - unambiguous, for developers"""
        return f"Point(x={self.x}, y={self.y})"
    
    def __str__(self):
        """User-friendly version"""
        return f"({self.x}, {self.y})"

point = Point(10, 20)

print(repr(point))  # Point(x=10, y=20) - unambiguous
print(str(point))   # (10, 20) - user-friendly
print(point)        # (10, 20) - uses __str__ if available

# In interactive Python:
# >>> point
# Point(x=10, y=20)  # Shows __repr__

# Ideal: repr() should return code that recreates the object
# eval(repr(point)) == point
```

---

### `__len__()` — Support len() Function

```python
class Playlist:
    def __init__(self, name, songs):
        self.name = name
        self.songs = songs
    
    def __len__(self):
        """Called by len() - return number of items"""
        return len(self.songs)
    
    def __repr__(self):
        return f"Playlist('{self.name}', {len(self)} songs)"

# Usage
playlist = Playlist("Summer Hits", ["Song1", "Song2", "Song3"])
print(len(playlist))  # 3 - calls __len__()
print(playlist)  # Playlist('Summer Hits', 3 songs)

# Without __len__, len(playlist) would raise:
# TypeError: object of type 'Playlist' has no len()
```

---

### `__getitem__()` — Support Indexing (obj[i])

```python
class Inventory:
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, index):
        """Called by obj[index] - support indexing"""
        if index < 0 or index >= len(self.items):
            raise IndexError("Inventory index out of range")
        return self.items[index]
    
    def __repr__(self):
        return f"Inventory({self.items})"

# Usage
inventory = Inventory(["Apple", "Banana", "Orange"])
print(inventory[0])  # Apple - calls __getitem__(0)
print(inventory[2])  # Orange
# print(inventory[5])  # IndexError

# Also support slicing!
class BetterInventory:
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, index):
        # Supports both single index and slicing
        if isinstance(index, slice):
            return self.items[index]  # Slicing
        else:
            return self.items[index]  # Single index

inv = BetterInventory(["A", "B", "C", "D"])
print(inv[1:3])  # ['B', 'C'] - slicing support!
```

---

### `__setitem__()` — Support Item Assignment

```python
class Matrix:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        """Get item at index"""
        return self.data[index]
    
    def __setitem__(self, index, value):
        """Called by obj[index] = value"""
        self.data[index] = value
    
    def __repr__(self):
        return f"Matrix({self.data})"

# Usage
matrix = Matrix([1, 2, 3, 4, 5])
print(matrix[0])  # 1

matrix[0] = 100  # Calls __setitem__(0, 100)
print(matrix)  # Matrix([100, 2, 3, 4, 5])

# Without __setitem__:
# matrix[0] = 100  # TypeError: 'Matrix' object does not support item assignment
```

---

### `__add__()`, `__sub__()`, etc. — Operator Overloading

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Called by v1 + v2 - addition operator"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Called by v1 - v2 - subtraction operator"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Called by v * 2 - multiplication operator"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

# Usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)

v3 = v1 + v2  # Calls __add__()
print(v3)  # Vector(4, 6)

v4 = v1 - v2  # Calls __sub__()
print(v4)  # Vector(-2, -2)

v5 = v1 * 3  # Calls __mul__()
print(v5)  # Vector(3, 6)

# Other operator dunders:
# __truediv__() for /
# __floordiv__() for //
# __mod__() for %
# __pow__() for **
# __and__() for &
# __or__() for |
# __xor__() for ^
```

---

### `__eq__()`, `__lt__()` — Comparison Operators

```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        """Called by obj1 == obj2 - equality"""
        if not isinstance(other, Student):
            return False
        return self.name == other.name and self.grade == other.grade
    
    def __lt__(self, other):
        """Called by obj1 < obj2 - less than"""
        return self.grade < other.grade
    
    def __le__(self, other):
        """Called by obj1 <= obj2 - less than or equal"""
        return self.grade <= other.grade
    
    def __gt__(self, other):
        """Called by obj1 > obj2 - greater than"""
        return self.grade > other.grade
    
    def __repr__(self):
        return f"Student('{self.name}', grade={self.grade})"

# Usage
s1 = Student("Alice", 95)
s2 = Student("Alice", 95)
s3 = Student("Bob", 87)

print(s1 == s2)  # True - calls __eq__()
print(s1 == s3)  # False

print(s1 > s3)  # True (95 > 87) - calls __gt__()
print(s3 < s1)  # True - calls __lt__()

# Sorting works with __lt__!
students = [s1, s3, Student("Carol", 92)]
sorted_students = sorted(students)
print(sorted_students)
# [Student('Bob', grade=87), Student('Carol', grade=92), Student('Alice', grade=95)]

# Other comparison dunders:
# __ne__() for !=
# __ge__() for >=
```

---

### `__call__()` — Make Object Callable Like a Function

```python
class Multiplier:
    """Object that acts like a function"""
    
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """Called when you use obj(args) - makes object callable"""
        return x * self.factor

# Usage
double = Multiplier(2)
triple = Multiplier(3)

print(double(5))  # 10 - calls __call__(5)
print(triple(5))  # 15

# Useful for decorators and callbacks
def apply_function(func, value):
    """Works with both functions and callable objects"""
    return func(value)

print(apply_function(double, 7))  # 14
print(apply_function(lambda x: x + 1, 7))  # 8

# Real-world example: Function wrapper
class Logger:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        print(f"Calling {self.func.__name__} with args={args}, kwargs={kwargs}")
        result = self.func(*args, **kwargs)
        print(f"Result: {result}")
        return result

@Logger
def add(a, b):
    return a + b

result = add(3, 5)
# Calling add with args=(3, 5), kwargs={}
# Result: 8
# 8
```

---

### `__enter__()` / `__exit__()` — Context Managers (with statement)

```python
class DatabaseConnection:
    """Context manager for safe resource management"""
    
    def __init__(self, filename):
        self.filename = filename
        self.connection = None
    
    def __enter__(self):
        """Called when entering 'with' block"""
        print(f"Opening connection to {self.filename}")
        self.connection = f"Connection to {self.filename}"
        return self.connection  # What's assigned to 'as' variable
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block (even if error occurs)"""
        print(f"Closing connection to {self.filename}")
        self.connection = None
        return False  # Don't suppress exceptions

# Usage
with DatabaseConnection("data.db") as conn:
    print(f"Using: {conn}")
    # Do database operations
# Output:
# Opening connection to data.db
# Using: Connection to data.db
# Closing connection to data.db

# Ensures cleanup even if error occurs
class SafeFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if exc_type:
            print(f"Error occurred: {exc_type.__name__}: {exc_val}")
        return False  # Propagate exception

# Usage
with SafeFile("test.txt", "w") as f:
    f.write("Hello, World!")
# File automatically closed after 'with' block

# Comparison with try/finally (verbose):
# try:
#     f = open("test.txt", "w")
#     f.write("Hello")
# finally:
#     f.close()
```

---

### `__iter__()` / `__next__()` — Iterator Protocol

```python
class CountUp:
    """Custom iterator that counts from 1 to n"""
    
    def __init__(self, max):
        self.max = max
        self.current = 0
    
    def __iter__(self):
        """Called by iter(obj) - return iterator object"""
        return self
    
    def __next__(self):
        """Called by next(obj) - return next item"""
        self.current += 1
        if self.current > self.max:
            raise StopIteration  # Signal end of iteration
        return self.current

# Usage
counter = CountUp(3)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3
# print(next(counter))  # StopIteration

# Works with for loops!
for num in CountUp(5):
    print(num)  # 1, 2, 3, 4, 5

# More practical example: Fibonacci sequence
class Fibonacci:
    def __init__(self, limit):
        self.limit = limit
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.a > self.limit:
            raise StopIteration
        result = self.a
        self.a, self.b = self.b, self.a + self.b
        return result

# Usage
fib_sequence = list(Fibonacci(50))
print(fib_sequence)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Or use generator function (simpler, but same result):
def fibonacci_gen(limit):
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b

print(list(fibonacci_gen(50)))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

---

### Complete Example: Custom List Class

```python
class CustomList:
    """Demonstrates multiple dunders working together"""
    
    def __init__(self, items=None):
        self.items = items if items else []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    
    def __add__(self, other):
        if isinstance(other, CustomList):
            return CustomList(self.items + other.items)
        return CustomList(self.items + [other])
    
    def __eq__(self, other):
        if isinstance(other, CustomList):
            return self.items == other.items
        return False
    
    def __iter__(self):
        return iter(self.items)
    
    def __repr__(self):
        return f"CustomList({self.items})"
    
    def __str__(self):
        return f"[{', '.join(map(str, self.items))}]"
    
    def __call__(self):
        return f"CustomList with {len(self)} items"

# Usage
lst1 = CustomList([1, 2, 3])
lst2 = CustomList([4, 5])

print(lst1)  # [1, 2, 3] - uses __str__()
print(repr(lst1))  # CustomList([1, 2, 3]) - uses __repr__()
print(len(lst1))  # 3 - uses __len__()
print(lst1[0])  # 1 - uses __getitem__()

lst1[0] = 10  # Uses __setitem__()
print(lst1)  # [10, 2, 3]

lst3 = lst1 + lst2  # Uses __add__()
print(lst3)  # [10, 2, 3, 4, 5]

print(lst1 == lst2)  # False - uses __eq__()

for item in lst1:  # Uses __iter__()
    print(item)  # 10, 2, 3

print(lst1())  # CustomList with 3 items - uses __call__()
```
Here is a practical example, given by **
```python
class Vault:
    def __init__(self, galleons=0, sickles=0, knuts=0):
        self.galleons = galleons
        self.sickles = sickles
        self.knuts = knuts

    def __str__(self): # determins what to print if print is called
        return f"{self.galleons} Galleons, {self.sickles} Sickles, {self.knuts} Knuts"

    def __add__(self, other): # overloads the + operator for Vault objects
        galleons = self.galleons + other.galleons
        sickles = self.sickles + other.sickles
        knuts = self.knuts + other.knuts
        return Vault(galleons, sickles, knuts)


potter = Vault(100, 50, 25)
print(potter)

weasley = Vault(25, 50, 100)
print(weasley)

total = potter + weasley
print(total)
```
---

## Summary

These 17 advanced topics represent deep knowledge of Python. Key takeaways:

1. **Multiprocessing** for true parallelism on CPU-bound work
2. **Descriptors & properties** for elegant attribute access
3. **Metaclasses** for framework-level control
4. **Slots** for memory optimization
5. **Context variables** for async context management
6. **Named tuples** for lightweight data structures
7. **Pickle** for object serialization
8. **Sockets** for network communication
9. **SQLAlchemy ORM** for database abstraction
10. **FastAPI** for modern async web APIs
11. **ML frameworks** for predictive modeling
12. **Visualization** for data exploration
13. **Package development** for code distribution
14. **Garbage collection** for memory efficiency
15. **Bytecode** for understanding execution
16. **C extensions** for performance-critical code
17. **Dunder methods** for customizing object behavior with operators and built-ins

Start with the basics, profile your code, and only optimize when necessary. Most Python code doesn't need these advanced techniques!

