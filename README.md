# 🗞️ Cross-National News Sentiment Analysis
### Greece 🇬🇷 × South Korea 🇰🇷 — Comparative Media Study

A end-to-end NLP pipeline that scrapes, translates, and analyses headline sentiment from two geographically distant news outlets, then compares how each country's media frames the news.

---

## Overview

This project asks a simple question: **do Greek and South Korean news headlines differ in tone, topic, and emotional framing?** To answer it, I built a pipeline that collects live headlines from two outlets, runs them through a series of NLP techniques, and produces a set of comparative visualisations.

| Outlet | Country | Section scraped |
|---|---|---|
| [To Vima](https://www.tovima.gr) | Greece 🇬🇷 | Society |
| [The Korea Herald](https://www.koreaherald.com) | South Korea 🇰🇷 | National |

---

## Pipeline

```
Web Scraping → Translation → Sentiment Analysis → Topic Modelling
→ Zero-Shot Classification → NER → Event Detection
→ Framing & Readability → Semantic Clustering
```

### 1. Web Scraping
Custom scrapers built with `requests` and `BeautifulSoup`. Greek headlines are extracted from image alt-text inside `<figure>` tags; Korean headlines from `<a href="/article/...">` links. Dates are parsed from adjacent HTML elements on the same listing page — no article-level fetching required.

### 2. Translation
Greek headlines are translated to English via `deep-translator` (Google Translate). The **original Greek text is preserved** alongside the translation so the sentiment model can score it natively, avoiding the translation bottleneck entirely.

### 3. Sentiment Analysis — XLM-RoBERTa + VADER
Two methods are run and compared:

- **XLM-RoBERTa** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) — a transformer fine-tuned on 198M multilingual tweets. Scores Greek headlines in their **original language**, eliminating translation artefacts. This is the primary signal used throughout.
- **VADER** — a rule-based English sentiment analyser applied to the translated headlines as a secondary baseline.

Comparing both methods on the same translated text gives a quantitative measure of how much sentiment is distorted by translation.

### 4. Exploratory Analysis
- Violin plots and stacked bar charts of sentiment distribution per country
- Weekly average sentiment time series
- Sentiment heatmap by day of week
- Top 5 most positive and negative headlines per outlet
- Interactive Folium geographic map

### 5. Topic Modelling — BERTopic
BERTopic clusters headlines by semantic meaning using sentence embeddings (`all-MiniLM-L6-v2`), rather than raw word counts. This means "PM resigns" and "Prime Minister steps down" land in the same cluster. Topics are labelled automatically with their most distinctive keywords via c-TF-IDF.

Alongside BERTopic, TF-IDF is used to surface the most **country-distinctive vocabulary** — terms that are common in one outlet but rare in the other.

### 6. Zero-Shot Topic Classification
Headlines are assigned to one of eight thematic categories (politics, crime, economy, health, military, society, environment, international relations) **without any labelled training data**, using `cross-encoder/nli-MiniLM2-L6-H768`. The model frames classification as natural language inference: *does this headline entail topic X?*

Headlines are classified in batches of 16 for efficiency (~6× faster than one-at-a-time).

### 7. Named Entity Recognition (NER)
spaCy (`en_core_web_sm`) extracts people, organisations, locations, and nationalities from each headline. Cross-referencing the entity lists between the two outlets identifies stories and figures that both countries covered independently.

### 8. Event Detection via Sentiment Spikes
Z-scores are computed **per country** on daily average sentiment. Days where `|z| > 1.5` are flagged as potential event-driven outliers, annotated on a time series, and the top headlines from those days are surfaced automatically.

### 9. Framing & Readability
- **NRC Emotion Lexicon** — maps words to 8 fine-grained emotions (anger, fear, joy, trust, sadness, disgust, anticipation, surprise), revealing each outlet's emotional palette beyond positive/negative.
- **Readability metrics** — Flesch-Kincaid grade level, average word length, and headline length, measuring the assumed reading level of each audience.

### 10. Semantic Clustering — UMAP + HDBSCAN
Headlines are embedded with a sentence transformer, reduced to 2D with UMAP (cosine metric), and clustered with HDBSCAN. Clusters shared between both countries reveal stories that both outlets covered; country-specific clusters reveal divergent agendas.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Scraping | `requests`, `beautifulsoup4` |
| Translation | `deep-translator` |
| Sentiment | `transformers` (XLM-RoBERTa), `vaderSentiment` |
| Topic modelling | `bertopic`, `sentence-transformers`, `scikit-learn` |
| Zero-shot NLI | `transformers` (MiniLM cross-encoder) |
| NER | `spacy` |
| Clustering | `umap-learn`, `hdbscan` |
| Readability | `textstat` |
| Visualisation | `plotly`, `matplotlib`, `seaborn`, `folium`, `wordcloud` |
| Data | `pandas`, `numpy`, `scipy` |

---

## Getting Started

### Requirements
Python 3.9+

### Install dependencies
```bash
pip install deep-translator vaderSentiment wordcloud plotly folium nltk scikit-learn \
            transformers torch bertopic sentence-transformers umap-learn hdbscan \
            spacy textstat requests beautifulsoup4
python -m spacy download en_core_web_sm
```

### Run
Open `news_sentiment_analysis.ipynb` in Jupyter and run all cells top to bottom.  
All installs are handled by `%pip` cells at the top of the notebook.

> **Note:** The zero-shot classification step (section 7) takes the longest — around 5–10 minutes on CPU depending on the number of headlines collected.

---

## Output Files

After running the notebook, the following CSVs are saved to the working directory:

| File | Contents |
|---|---|
| `news_sentiment_combined.csv` | Full enriched dataset with all sentiment, topic, and entity columns |
| `greek_news_clean.csv` | Cleaned Greek headlines only |
| `korea_news_clean.csv` | Cleaned Korean headlines only |
| `semantic_clusters.csv` | Headlines with UMAP coordinates and cluster assignments |

---

## Methodological Notes

- **Why XLM-RoBERTa over VADER?** VADER was designed for English social media. Applying it to machine-translated Greek introduces systematic noise — word choice shifts, negations are sometimes lost, and idioms rarely survive. XLM-RoBERTa processes each language natively, making sentiment scores genuinely cross-linguistically comparable.
- **Why keep VADER at all?** Running both on translated text and measuring their agreement rate gives a concrete, quantifiable estimate of translation-induced sentiment distortion — which is itself an interesting finding.
- **Why BERTopic over LDA?** LDA treats headlines as bags of words; BERTopic uses sentence embeddings that capture semantic relationships. For short texts like headlines, this makes a material difference in cluster coherence.
- **Dates with no parse match** are kept as `NaT` rather than dropped, so every headline contributes to sentiment and topic analysis even when the time series can't include it.

---

## Project Status

This is a portfolio project built to practise applied NLP on real-world data. The scrapers target live pages and may break if the sites restructure their HTML.

---

*Built with Python · Data collected from public news listing pages for academic purposes*
