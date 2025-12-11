# DedupKit

Semantic deduplication using embeddings.

## Installation
```bash
pip install dedupkit
```

## Quick Start
```python
from dedupkit import Deduplicator
from dedupkit.providers import LocalEmbeddingProvider
from dedupkit.storage import MemoryStorage

dedup = Deduplicator(
    embedding=LocalEmbeddingProvider(),
    storage=MemoryStorage(),
    threshold=0.85
)

# Add items
dedup.add("Login button is broken", item_id="BUG-001")
dedup.add("Payment form crashes", item_id="BUG-002")

# Check for duplicates
result = dedup.check("Login button not working")
print(result.is_duplicate)      # True
print(result.matches[0].id)     # BUG-001
print(result.matches[0].similarity)  # 0.92
```

## Providers

**Local (no API key needed):**
```python
from dedupkit.providers import LocalEmbeddingProvider
provider = LocalEmbeddingProvider()
```

**OpenAI:**
```python
from dedupkit.providers import OpenAIEmbeddingProvider
provider = OpenAIEmbeddingProvider(api_key="sk-...")
```

## License

MIT
