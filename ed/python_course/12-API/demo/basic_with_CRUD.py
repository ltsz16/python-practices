from fastapi import FastAPI
from pydantic import BaseModel

# Create a FastAPI application instance
app = FastAPI()

# Define a data model
class Item(BaseModel):
    id: int
    name: str
    description: str = None
    price: float = None

# In-memory storage (for demo purposes)
items_db = {}

# GET all items
@app.get("/items")
def get_all_items():
    """Get all items."""
    return {"items": items_db}

# GET a specific item by ID
@app.get("/items/{item_id}")
def get_item(item_id: int):
    """Get a specific item by ID."""
    if item_id in items_db:
        return {"item": items_db[item_id]}
    return {"error": "Item not found"}

# POST - Create a new item
@app.post("/items")
def create_item(item: Item):
    """Create a new item."""
    items_db[item.id] = item
    return {"message": "Item created", "item": item}

# PUT - Update an existing item
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    """Update an existing item."""
    if item_id in items_db:
        items_db[item_id] = item
        return {"message": "Item updated", "item": item}
    return {"error": "Item not found"}

# DELETE - Delete an item
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    """Delete an item by ID."""
    if item_id in items_db:
        deleted_item = items_db.pop(item_id)
        return {"message": "Item deleted", "item": deleted_item}
    return {"error": "Item not found"}

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint that returns a hello world message."""
    return {"message": "Hello, World!"}

# Run standalone
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# POST - Create
# curl -X POST "http://127.0.0.1:8000/items" -H "Content-Type: application/json" -d '{"id": 1, "name": "Widget", "price": 9.99}'

# GET - Retrieve
# curl "http://127.0.0.1:8000/items/1"

# PUT - Update
# curl -X PUT "http://127.0.0.1:8000/items/1" -H "Content-Type: application/json" -d '{"id": 1, "name": "Updated Widget", "price": 12.99}'

# DELETE - Remove
# curl -X DELETE "http://127.0.0.1:8000/items/1"