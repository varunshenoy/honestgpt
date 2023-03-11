# Building a Better ChatGPT
## Grounding LLMs in Truth With Under 30 Lines of Code

This demo is part of a presentation at an SF Python meetup in March 2023. The slides are also in this repo.

### Requirements

Run `pip3 install -r requirements.txt` in the root directory of the project.

This will install the following packages:
```
chromadb==0.3.11
langchain==0.0.107
numpy==1.24.2
openai==0.26.4
sentence_transformers==2.2.2
```

## Part 1: Manual Embedding and Generation

**Code:** `basic.py`

Leverage [Sentence Transformer embeddings](https://www.sbert.net/) and basic matrix multiplication to identify the parts of a document most relevent to a user query. These parts will then be carefully crafted into a prompt, along with the user's query, before being sent to the OpenAI GPT-3 API.

The result will be a coherent summary that contains citations to segments in our original document, resulting in a shorter and more precise summary than ChatGPT that is also grounded in truth.

**Line total:** ~120

## Part 2: Chroma

**Code:** `using_chromadb.py`

Lots of abstractions have been built out for standard language model operations. We will replace our hand-rolled Sentence Transformer + Numpy approach with [Chroma](https://www.trychroma.com/), an open-source embedding database that can run locally.

Our result will be similar to the one above.

**Line total:** ~100

## Part 3: Langchain

**Code:** `using_langchain.py`

We will compress the amount of code we've written even further by using [Langchain](https://langchain.readthedocs.io/en/latest/) and its `load_qa_with_sources_chain`. We will still use Chroma, but with [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) instead.

While this is the simplest solution with the least amount of code, it is much harder to edit the prompt and have fine-grained control over the embeddings part.

**Line total:** ~30

## Extensions
- Implement a chat interface and incorporate chat history into the prompt
- Try out other types of embeddings or a vector database
- Test out the code on different text files, or more complex documents using [Langchain's dataloaders](https://langchain.readthedocs.io/en/latest/modules/document_loaders.html).
- Play with other approximate nearest neighbors libraries, like FAISS, annoy, or hnswlib.