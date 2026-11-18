**Major Versioning Changes**
* Remove or rename APIs or API params
* Changes in the behavior of an existing API
* Changes in error codes and fault contracts
* New features
* Anything that violates the Principle of Least Astonishment

API versioning is Major.Minor only

**Deprecation**
```python
@app.get(
    "/old-endpoint",
    deprecated=True,
    description="This endpoint is deprecated as of v2.0. Please use /new-endpoint instead."
)
async def old_endpoint():
    return {"message": "This endpoint is deprecated"}
```