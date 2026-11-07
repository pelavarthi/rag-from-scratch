# Retrieval-Augmented Generation

RAG grounds a language model in your own documents. Instead of asking the
model to answer from memory, you search your documents for the passages
relevant to a question and hand those passages to the model along with the
question.

## What is RAG?

A RAG system has three parts. First, an ingestion step loads documents and
splits them into chunks small enough to embed and retrieve. Second, an
embedding step turns each chunk into a vector, a list of numbers that
captures what the chunk means. Third, at query time, the question is embedded
the same way, the nearest chunk vectors are found, and those chunks become
the context for the answer.

The key insight is that retrieval and generation are separate jobs. Retrieval
finds facts. Generation turns facts into a fluent answer with citations. When
the facts are wrong or missing, the answer is wrong, no matter how good the
language model is. That is why chunking quality matters so much: a chunk that
cuts a fact in half can hide it from the search entirely.
