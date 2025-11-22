# Medical Audit Service

FastAPI service for auditing face sheets and wound notes. Includes optional
AWS Textract OCR support and a Lambda-ready entrypoint so the app can run in
serverless or containerized environments.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints:

- `GET /health` – health probe
- `GET /medical/config` – audit configuration metadata
- `POST /medical/audit/face-sheet` – audit a face-sheet PDF
- `POST /medical/audit/wound-note` – audit a wound-note PDF
- `POST /medical/audit/auto` – auto-detect document type and audit
- `POST /vision/motion/detect` – detect human-motion segments in uploaded video
- `GET /vision/motion/config` – tunable parameters for the motion detector
- `POST /vision/swing/analyze` – analyze golf-swing tempo and posture using pose estimation
- `GET /vision/swing/config` – swing-analyzer input/output contract

## AWS readiness

- The project ships with a Dockerfile suitable for ECS/App Runner or Lambda
  container images. Build with `docker build -t medical-audit .`.
- A Mangum handler is exposed in `app.main:handler` for Lambda runtimes.
- Optional OCR for image-only PDFs is available via Amazon Textract. Configure
  through environment variables:

  - `MED_AUDIT_TEXTRACT_ENABLED=true`
  - `MED_AUDIT_AWS_REGION` (defaults to `us-east-1`)
  - `MED_AUDIT_MAX_PDF_PAGES` (to cap local processing per request)

If Textract is disabled, the service still runs normally and responds with a
clear message when a scanned PDF needs OCR.

