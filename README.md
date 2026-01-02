# Misinformation Claim Verification System

An experimental end-to-end system for analyzing textual claims using
retrieval-augmented pipelines, reranking, and Transformer based reasoning.

The project focuses on system design, robustness, and real world constraints
rather than building a definitive fact-checking oracle.


## Overview

The system verifies claims by:
1. Retrieving relevant evidence using semantic search (FAISS)
2. Reranking evidence using a cross-encoder
3. Predicting claim stance using a Natural Language Inference model
4. Generating concise explanations using a summarization model

External evidence sources are optionally integrated when available, with
graceful degradation when APIs are unavailable.


## Core Components

### Text Pipeline
- Semantic retrieval using Sentence Transformers + FAISS
- Cross-encoder reranking for relevance refinement
- NLI-based classification (entailment / neutral / contradiction)
- Evidence-aware summarization

### External Evidence 
- Google Fact Check API
- NewsAPI
- Web fallback via controlled scraping

### Image Verification (Implemented, Disabled in Demo)
- AI-generated image detection
- Classical image manipulation analysis (ELA)
- OCR and reverse image search using Google Vision API  
Image pipelines are gated due to external API costs.


## Models Used

- **Retrieval:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Claim Classification:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- **Summarization:** `google/flan-t5-large`


## Interfaces

- **Gradio UI:** Public interactive demo (Hugging Face Spaces)
- **FastAPI Backend:** Included to demonstrate API-based deployment (not public)


## Deployment

- CPU-only inference
- Deployed as a Gradio app on Hugging Face Spaces
- Large artifacts (FAISS index, corpora, checkpoints) are not included in this repository


## Limitations

- Designed as an experimental research system
- Claims with strong contextual ambiguity may be misclassified
- Image pipeline disabled in public demo


## Notes on Data

A cleaned notebook demonstrating corpus construction using web scraping and
external APIs is included. Generated artifacts are not committed.


## Demo

👉 Hugging Face Space: https://huggingface.co/spaces/Sudh1853/Claim_verification

