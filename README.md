# Smart Retail & Customer Intelligence Platform

AI-powered retail backend that recognizes returning customers via face recognition,
classifies product images, analyzes review/chat sentiment, and answers FAQs through a
chatbot — all behind one FastAPI service.

Internship project — **Ayush**, Computer Science & Engineering, VIT Bhopal University.

## Project Structure

```
smart-retail-ai/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # Settings (API key, model/data paths)
│   ├── security.py          # API key auth dependency
│   ├── schemas.py           # Pydantic request/response models
│   ├── routers/              # vision.py, nlp.py, chatbot.py, dashboard.py
│   ├── services/              # cv_service, nlp_service, chatbot_service, cv_utils
│   └── models/               # trained artifacts (generated — not in git)
├── notebooks/                # training notebooks, run once before starting the API
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_face_recognition_setup.ipynb
│   └── 03_sentiment_model_training.ipynb
├── data/
│   ├── intents.json          # chatbot FAQ intents
│   └── reviews_sample.csv    # demo reviews dataset (swap with Kaggle data)
├── tests/
│   └── test_endpoints.py
├── requirements.txt
├── Dockerfile
└── .github/workflows/deploy.yml
```

## 1. Setup (VS Code)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # edit API_KEY if you want something other than the default
```

In VS Code: install the **Python** and **Jupyter** extensions, then select the `.venv`
interpreter (bottom-right or `Ctrl+Shift+P` → "Python: Select Interpreter").

## 2. Train the models

Open each notebook in `notebooks/` and run all cells, in order:

1. `01_image_classifier_training.ipynb` → `app/models/product_classifier.h5` + `class_names.json`
2. `02_face_recognition_setup.ipynb` → `app/models/face_recognizer.yml` + `face_db.pkl`
3. `03_sentiment_model_training.ipynb` → `app/models/sentiment_model.pkl`, `vectorizer.pkl`,
   `chatbot_model.pkl`, `chatbot_vectorizer.pkl`

These use small stand-in datasets (CIFAR-10, an LFW subset, and a 20-row sample review set) so
everything trains in a couple of minutes. Swap in your real retail images / consented customer
photos / Kaggle reviews CSV the same way — the rest of the pipeline doesn't change.

## 3. Run the API

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger docs. Every endpoint needs an
`X-API-Key` header matching `API_KEY` in `.env` (default: `demo-secret-key`).

| Endpoint | Method | Purpose |
|---|---|---|
| `/classify-product` | POST (multipart file) | Predicts product category from an image |
| `/recognize-face` | POST (multipart file) | Recognizes a returning customer / logs a visit |
| `/analyze-sentiment` | POST (JSON `{"text": ...}`) | Positive/negative sentiment + confidence |
| `/chatbot` | POST (JSON `{"message": ...}`) | FAQ chatbot reply |
| `/dashboard/stats` | GET | Aggregate visit/sentiment stats |

## 4. Run tests

```bash
pytest -q
```

The test suite mocks the ML services, so it passes even before you've trained any models —
useful for CI. Run it again after training for a full integration pass.

## 5. Docker

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 --env-file .env -v $(pwd)/app/models:/code/app/models smart-retail-ai
```

The `-v` volume mount makes your locally trained model files available inside the container
without baking them into the image.

## 6. Deploy

Push to a GitHub repo — `.github/workflows/deploy.yml` runs lint + tests + a Docker build on
every push to `main`. From there, deploy the image to Render, Railway, AWS EC2, or Google Cloud
Run (all have student/free tiers).

## Ethics & Privacy Note

Face recognition requires explicit customer consent, secure storage of face encodings, a data
retention limit, and bias testing across demographic groups before any real deployment — see
Section 6 of the project report for the full discussion.
