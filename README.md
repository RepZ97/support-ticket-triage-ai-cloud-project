# Support Ticket Triage

An end-to-end AI application that reads a consumer finance complaint, routes it
to the right product queue with a trained classifier, and then uses Gemini on
Vertex AI to judge urgency and draft a reply for the agent.

**Live deployment:** <https://ticket-triage-616119747229.us-central1.run.app>

---

## 1. Problem Statement

Financial services firms receive complaints as unstructured free text through
web forms, email and call-centre notes. Before anyone can act on one, two
questions have to be answered:

1. **Which team owns this?** A mortgage escrow dispute and a debt-collection
   harassment claim go to different specialists with different regulatory
   obligations.
2. **How fast does it need attention?** A complaint mentioning imminent
   foreclosure cannot sit in the same queue position as a request for an old
   statement.

Doing this by hand is slow, inconsistent between staff, and scales badly.
Misrouted complaints get bounced between teams, and genuinely urgent cases are
easy to lose in the volume.

## 2. Use Case

The application sits at the intake point of a support desk. An agent (or an
upstream automation) submits the complaint text and immediately receives:

- the product queue it should be routed to, with a confidence score and the
  next-best alternatives,
- an urgency rating with a one-line justification,
- a short summary, and
- a draft reply the agent can edit and send.

It is usable directly through the web page, or as a JSON API called by an
existing ticketing system.

## 3. Solution Overview

The two questions above have very different characters, so the system uses a
different technique for each rather than forcing one model to do both.

**Routing** is a supervised classification problem with a large volume of real
labelled data behind it, so it is solved with a trained classifier. This is
fast, free to run, deterministic, and can be measured against a held-out test
set.

**Urgency and reply drafting** have no ground-truth labels in the dataset, and
inventing a proxy label would not be honest. These are judgement tasks, so they
are handled zero-shot by a large language model given the complaint plus the
category the classifier already assigned.

The two stages are independent by design. If the LLM layer is unavailable, the
API still returns routing, flags why the assessment is missing, and the UI shows
the routing result with a warning. The core function never depends on the
optional one.

## 4. Dataset

**Source:** [`milesbutler/consumer_complaints`](https://huggingface.co/datasets/milesbutler/consumer_complaints)
on Hugging Face — a published extract of the
[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
the US Consumer Financial Protection Bureau's public record of complaints filed
against financial companies.

The extract contains 208,398 complaints that include a narrative, i.e. those
where the consumer consented to publication of the text. Two columns are used:
`Consumer Complaint` (the narrative) and `Product` (the category, used as the
routing label).

`training/prepare_data.py` downloads it and applies the following, all of which
were driven by what the raw data actually looks like:

| Step | Rows |
|---|---|
| Raw extract | 208,398 |
| Drop empty / very short narratives (< 40 chars) | 207,658 |
| Map product labels to a canonical set | 207,437 |
| Remove duplicate narratives | **201,679** |

Two properties of this data needed handling:

**The labels overlap.** CFPB renamed its product categories several times over
the life of the database, so the raw data carries 18 labels for what are really
8 products — `Credit reporting` and `Credit reporting, credit repair services,
or other personal consumer reports` are the same queue, as are `Credit card`,
`Prepaid card` and `Credit card or prepaid card`. Training on the raw labels
would mean penalising the model for confusing two names for one thing. They are
collapsed into 8 canonical classes. `Other financial service` (221 rows) is
dropped: it is a catch-all bucket, too small and too semantically mixed to
learn from.

**The text is redacted.** CFPB replaces names, dates and account numbers with
runs of `X` before publishing. Left alone, `xxxx` becomes one of the highest
weighted features in the vocabulary while carrying no signal, so the redaction
markers are added to the stop-word list.

Final class distribution (201,679 rows, split 80/20 stratified):

| Category | Rows |
|---|---|
| Credit reporting | 55,313 |
| Debt collection | 46,520 |
| Mortgage | 32,771 |
| Credit or prepaid card | 23,125 |
| Bank account | 15,918 |
| Student loan | 12,417 |
| Consumer loan | 12,158 |
| Money transfer / virtual currency | 3,457 |

## 5. AI/ML Approach

### Routing classifier

TF-IDF features into a linear classifier — a strong, well-understood baseline
for topic classification on medium-length text, and cheap enough to run
in-process on every request with no GPU.

- **Features:** `TfidfVectorizer`, word unigrams + bigrams, `min_df=3`,
  capped at 100,000 features, sublinear term frequency, English stop words plus
  the CFPB redaction markers.
- **Split:** 80/20 stratified, `random_state=42`, giving 161,343 training and
  40,336 test rows.

Two classifiers were trained and compared on the held-out test set:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| LinearSVC (`C=0.5`) | 0.8636 | 0.8392 |
| **Logistic Regression (`C=4.0`)** — served | 0.8599 | 0.8353 |

LinearSVC scores marginally higher, but it only exposes `decision_function`,
not calibrated class probabilities. The API returns a confidence score that the
UI displays and that a downstream system could threshold on, so a real
probability is worth more here than 0.4 percentage points of accuracy.
Logistic regression is what ships; the SVC number is kept in `/model-info` as
the reported baseline.

Per-class results for the served model:

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Mortgage | 0.929 | 0.945 | 0.937 | 6,554 |
| Student loan | 0.909 | 0.873 | 0.891 | 2,483 |
| Credit reporting | 0.880 | 0.876 | 0.878 | 11,063 |
| Debt collection | 0.833 | 0.866 | 0.849 | 9,304 |
| Bank account | 0.822 | 0.827 | 0.824 | 3,184 |
| Credit or prepaid card | 0.812 | 0.824 | 0.818 | 4,625 |
| Money transfer / virtual currency | 0.856 | 0.687 | 0.762 | 691 |
| Consumer loan | 0.769 | 0.682 | 0.723 | 2,432 |
| **Accuracy** | | | **0.860** | 40,336 |
| **Macro avg** | 0.851 | 0.822 | 0.835 | 40,336 |

The weakest class is `Consumer loan`, which is expected: it is the merged
bucket of four legacy labels (vehicle, payday, personal, generic consumer
loans) and overlaps genuinely with `Debt collection` and `Credit or prepaid
card`. `Money transfer / virtual currency` has good precision but weak recall,
which is the normal consequence of being the smallest class at 1.7% of the
data. Confusion is concentrated between semantically adjacent queues rather
than scattered, which is the behaviour you want — a misroute lands somewhere
plausible.

### Urgency and reply drafting

Gemini (`gemini-2.5-flash` by default) on Vertex AI, called with a system
instruction that defines the three urgency levels in concrete terms (money
currently inaccessible or foreclosure imminent → `high`; real but not
time-critical impact → `medium`; information requests and historical disputes →
`low`). It is given the complaint and the classifier's category.

Output is constrained to a Pydantic schema via structured JSON output, so the
response is always parseable rather than free prose that needs regex. The
instruction explicitly forbids promising any outcome, refund or timeline, and
forbids repeating or inventing the `XXXX` redactions.

## 6. Application Architecture

```
Browser ── single static page (app/static/index.html)
   │
   ▼
FastAPI (app/main.py)
   │
   ├── POST /triage
   │      ├── app/classifier.py ── TF-IDF + Logistic Regression   (in-process, ~5 ms)
   │      │                         → category, confidence, alternatives
   │      └── app/llm.py ────────── Vertex AI Gemini              (network, ~1-3 s)
   │                                → urgency, reason, summary, draft reply
   │                                → optional; failure is reported, not fatal
   │
   ├── GET /health      liveness + whether the LLM layer is configured
   └── GET /model-info  served model, metrics, per-class scores, confusion matrix
```

Design decisions worth noting:

- **The model artifact is baked into the image**, not fetched from storage at
  boot. At 7.5 MB this costs nothing in image size and removes a network
  dependency plus an IAM permission from the cold-start path.
- **The classifier is loaded once at startup** via the FastAPI lifespan hook, so
  the first real request does not pay the deserialisation cost.
- **No API keys anywhere.** Vertex AI is reached with Application Default
  Credentials through the Cloud Run service account, so there is no secret to
  store, rotate or leak into the repository.

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn 1.9, joblib |
| Data prep | pandas, pyarrow |
| API | FastAPI, Uvicorn, Pydantic |
| LLM | Gemini via Vertex AI (`google-genai` SDK) |
| Frontend | Single HTML page, no framework or build step |
| Container | Docker, `python:3.12-slim` |
| Cloud | Google Cloud Run, Cloud Build, Artifact Registry, Vertex AI |

Cloud Run was chosen because this is a stateless request/response service with
bursty, unpredictable traffic. It scales to zero between requests, so an idle
demo deployment costs nothing, and it removes any cluster or VM management.

## 8. Local Setup Instructions

Requires Python 3.12 (via conda or otherwise).

```bash
conda create -n ticket-triage python=3.12 pip -y
conda activate ticket-triage
pip install -r requirements-dev.txt
```

Build the dataset and train the model. The first run downloads ~170 MB and
caches it in `data/`; training takes a few minutes.

```bash
python training/prepare_data.py
python training/train.py
```

Run the API:

```bash
uvicorn app.main:app --reload --port 8080
```

Open <http://localhost:8080>. Interactive API docs are at `/docs`.

The Gemini layer is optional locally. Without credentials the app runs fine and
reports the assessment as unavailable. To enable it:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Port to serve on (set automatically by Cloud Run) |
| `GOOGLE_CLOUD_PROJECT` | _unset_ | GCP project for Vertex AI; unset disables the LLM layer |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used for assessment |

## 9. Deployment Details

Deployed to **Google Cloud Run**, built with **Cloud Build**, image stored in
**Artifact Registry**, LLM served by **Vertex AI**.

Set your project and enable the APIs:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com aiplatform.googleapis.com
```

Grant the service account two roles before deploying. Cloud Run and Cloud Build
both run as the Compute Engine default service account unless told otherwise,
and on projects created recently that account starts with no roles at all:

```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# lets the deployed service call Gemini
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:${SA}" --role="roles/aiplatform.user"

# lets Cloud Build read the uploaded source and push the image
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:${SA}" --role="roles/cloudbuild.builds.builder"
```

The second role is easy to miss. Older projects granted the default service
account the broad Editor role, so source deploys worked without it; newer ones
do not, and `gcloud run deploy --source` fails with
`403 ... does not have storage.objects.get access` on the
`run-sources-*` bucket, which reads like a bucket problem rather than a missing
role.

Then deploy from source — Cloud Build builds the Dockerfile, pushes to Artifact
Registry and rolls out the revision in one step:

```bash
gcloud run deploy ticket-triage \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 120 \
    --max-instances 3 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1
```

`.gcloudignore` keeps the cached dataset (~233 MB) and the training code out of
the build upload, so only the app and the serialised model are sent.

Confirm it came up, including whether the LLM layer is active:

```bash
curl https://YOUR_SERVICE_URL/health
```

`"assessment": {"enabled": true, ...}` means Vertex AI is reachable. If it
reports `enabled: false`, the routing API is still fully functional — check the
`reason` field, which is usually a missing env var or the IAM binding above not
having propagated yet.

**Note on memory:** 1 GiB is specified because scikit-learn plus the vectoriser
vocabulary exceeds the 512 MiB default at load.

## 10. API / Web Application Usage

### Web interface

Open the service root. Paste a complaint (or use **Load sample**) and press
**Triage**. The urgency and draft reply can be switched off to get a
classifier-only response, which returns in milliseconds.

### `POST /triage`

```bash
curl -X POST https://YOUR_SERVICE_URL/triage \
  -H "Content-Type: application/json" \
  -d '{"text": "A debt collector keeps calling me about an account that is not mine. I sent a written request for validation over 45 days ago and they never responded, but they keep calling my workplace.", "include_assessment": true}'
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `text` | string | required | 20–20,000 characters |
| `include_assessment` | bool | `true` | `false` skips the LLM call |

```json
{
  "routing": {
    "category": "Debt collection",
    "confidence": 0.9634,
    "alternatives": [
      { "category": "Consumer loan", "confidence": 0.0273 },
      { "category": "Student loan", "confidence": 0.0048 }
    ]
  },
  "assessment": {
    "urgency": "medium",
    "urgency_reason": "Repeated workplace contact after an ignored validation request.",
    "summary": "Consumer disputes a debt and reports continued collection calls at work.",
    "draft_reply": "Thank you for contacting us about these collection calls..."
  },
  "assessment_error": null
}
```

If the LLM layer is unavailable, `assessment` is `null` and `assessment_error`
explains why. `routing` is always present.

### `GET /health`

Liveness, model training timestamp, and whether the assessment layer is
configured.

### `GET /model-info`

Served model, both candidates' scores, class list, vectoriser settings, full
per-class metrics and the confusion matrix.

### `GET /docs`

Generated OpenAPI documentation.

## 11. Docker Instructions

The image expects `models/classifier.joblib` to exist, so train before building
(see Local Setup). Training dependencies are not installed in the image — only
the serialised artifact ships.

```bash
docker build -t ticket-triage:local .
docker run --rm -p 8080:8080 ticket-triage:local
```

Then open <http://localhost:8080>.

To enable the Gemini layer in a local container, pass your project and mount
your Application Default Credentials read-only:

```bash
docker run --rm -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  ticket-triage:local
```

The container listens on `$PORT` (default `8080`) and runs as a non-root user.

## Repository Layout

```
app/
  main.py            FastAPI routes
  classifier.py      model loading and prediction
  llm.py             Vertex AI Gemini assessment
  static/index.html  web interface
training/
  prepare_data.py    download, clean, label-map, split
  train.py           train, compare, evaluate, serialise
models/
  classifier.joblib  trained pipeline (committed, 7.5 MB)
  metadata.json      metrics and configuration
Dockerfile
.dockerignore          what stays out of the image
.gcloudignore          what stays out of the Cloud Build upload
requirements.txt       runtime dependencies
requirements-dev.txt   adds training dependencies
environment.txt        conda environment definition
```

`data/` is generated by `prepare_data.py` and deliberately not committed.
