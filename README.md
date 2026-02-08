# Book Glancer  
*A lightweight NLP playground for quick literary insights (and conversations with characters from the novel!).*

Book Glancer is a hobby / portfolio project for exploring classic NLP techniques on book-length texts, with a simple interactive interface for “glancing” at a book and its translation through statistics, frequency analysis, and experimental LLM-based features.

The goal is not deep literary criticism, but fast, visual, and reproducible text exploration — plus the occasional opportunity to interview a character about their own story.

---

## What does it do?

Book Glancer lets you upload a book (PDF or TXT) and optionally its translation, then applies a set of NLP workflows.

### Core features
- Random mid-text sampling (jump straight into the book)
- Basic text statistics (words, characters, estimated pages)
- Tokenization, lemmatization, stopword filtering
- Word frequency distributions
- Bigram (2-gram) phrase extraction
- Word clouds for terms and phrases
- Named Entity Recognition (people, places, organizations)

### Experimental (local only)
These features combine classic NLP with local LLMs:

- LLM-based Q&A over the full book text  
- Character “medallion” generation  
  (NLP entity extraction → contextual text chunks → LLM summarization)  
- **Chat with book characters**  
  Characters are reconstructed from:
  - their extracted appearances in the text  
  - surrounding narrative context  
  - their detected dialogue style  

In practice, this means you can literally ask a character what they think about the plot so far.

Default demo text: *R.U.R. by Karel Čapek* (public domain).

---

## Local-first by design

Book Glancer is intentionally built as a **fully local application**:

- All text processing runs on your machine  
- All LLM interactions run via local Ollama  
- No book content, prompts, or conversations are sent to any external API  
- No cloud services, no telemetry, no data collection  

If you care about privacy, offline work, or experimenting with copyrighted texts, this setup is deliberate.

---

## LLM setup and technical limits

### Ollama requirement
The LLM features require **Ollama installed locally**.

By default, the project uses:

gemma2:2b
https://ollama.com/library/gemma2:2b
2.61B parameters, 1.6 GB

This model was chosen because it is:
- small enough to run on consumer hardware  
- fast for interactive use  
- good enough for summarization and role-style dialogue  

You can swap the model in the code for any other Ollama-supported model.

### Hosted vs local mode
If the app is running in a hosted environment (e.g. Streamlit Cloud):

- LLM features are **disabled**
- Only classic NLP features are available
- This is a hard technical limit: Ollama requires local execution

### Practical limitations
- Performance depends on your hardware  
- Large books may hit context or memory limits  
- Character chats are bounded by:
  - extracted text windows  
  - LLM context size  
  - model capabilities  

This is closer to an experimental NLP/LLM lab than a production-grade RAG system.

---

## Tech Stack

- Frontend: Streamlit  
- NLP: NLTK, spaCy  
- Data & Visualization: pandas, matplotlib, seaborn, wordcloud  
- Document handling: PyPDF2, filetype  
- LLM (local): Ollama (`gemma2:2b` by default)

Runs fully offline.

---

## Project Status

Work in progress and intentionally experimental.

Some features are prototypes, some are rough, and many are designed more for exploration than production use. The code favors clarity and hackability over strict optimization.

---

## Limitations & Notes

- Not intended for commercial use  
- No guarantee of linguistic or literary accuracy  
- LLM features require local Ollama installation  
- Character chats are approximations, not canonical interpretations  
- This is a learning project, not a literary authority  

---

## Summary

Book Glancer is a small NLP lab for books:  
upload a text, run classic NLP methods, visualize patterns, and — entirely offline — have a conversation with the people living inside the novel.