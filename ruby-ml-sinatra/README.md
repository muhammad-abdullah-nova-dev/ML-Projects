# Sentiment Analysis Blog Post Filter (Ruby Sinatra + FastAPI ML Backend)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Ruby](https://img.shields.io/badge/Ruby-3.0+-CC342D.svg?style=flat&logo=ruby&logoColor=white)](https://www.ruby-lang.org)
[![Sinatra](https://img.shields.io/badge/Sinatra-Web%20App-d35400.svg)](http://sinatrarb.com)

A cross-language web application demonstrating integration between a lightweight Ruby Sinatra frontend and a Python FastAPI machine learning microservice for real-time text sentiment classification.

Maintained and documented by **M. Abdullah**.

---

## 🏗️ Architecture

* **Frontend**: Ruby Sinatra web application handling user blog submissions and rendering posts with sentiment badges.
* **ML Service**: Python FastAPI microservice utilizing NLP sentiment polarity analysis (positive, neutral, negative).
* **Communication**: Asynchronous HTTP JSON requests across microservice boundaries.

---

## ⚡ Getting Started

### 1. Python ML Microservice
```bash
cd ruby-ml-sinatra/ml_api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Ruby Sinatra Application
```bash
cd ruby-ml-sinatra/ruby_app
bundle install
ruby app.rb -p 4567
```
Visit `http://localhost:4567` to submit blog posts and view automatic sentiment classification.

---

## 👨‍💻 Maintainer & Attribution
- **Maintainer**: **M. Abdullah**
- **Original Project Origin**: Derivative work based on open-source project by `torresjchristopher`.
- **Copyright**: Copyright © 2026 M. Abdullah for modifications.
