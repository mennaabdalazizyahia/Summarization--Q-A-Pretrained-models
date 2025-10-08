# 📝 Text Summarization & Question Answering App  
Built with **Streamlit** and **Hugging Face Transformers**

## 📖 Overview  
This project is a simple web application that performs two main NLP tasks:

1. **Text Summarization** – Generate concise summaries from long articles or documents.  
2. **Question Answering (Q/A)** – Answer user questions based on a given text context.  

The app leverages pre-trained models from the **Hugging Face Transformers** library:
- `facebook/bart-large-cnn` → for text summarization  
- `deepset/roberta-base-squad2` → for question answering  

---

## ⚙️ Requirements  

Before running the app, make sure you have the following packages installed:

```bash
pip install streamlit transformers torch
