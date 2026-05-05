# SentiLens — Sentiment Analysis Web App

A professional full-stack sentiment analysis web app built for CSC 309.

## Deploy to Render (3 steps)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/sentilens.git
   git push -u origin main
   ```

2. **Create Web Service on Render**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repo
   - Render auto-detects `render.yaml` — just click **Deploy**

3. **Done.** Your app is live at `https://sentilens.onrender.com`

---

## Run Locally

```bash
pip install -r requirements.txt
python build.py        # trains the model once
python app.py          # starts at http://localhost:5000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| POST | `/api/predict` | Single prediction |
| POST | `/api/predict/batch` | Batch (up to 100) |
| GET | `/api/health` | Model status |

## Project Structure

```
sentimentapp/
├── app.py              # Flask server
├── build.py            # Trains model at deploy time
├── templates/
│   └── index.html      # Frontend UI
├── src/
│   ├── preprocessor.py
│   ├── dataset.py
│   ├── train.py
│   └── inference.py
├── models/             # Auto-created on first run
├── requirements.txt
├── render.yaml
└── Procfile
```
