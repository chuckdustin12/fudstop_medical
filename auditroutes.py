import logging
import re
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader

from app.aws_utils import textract_document_text
from app.settings import settings

# Silence the noisy PyPDF2 cmap warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2._cmap")

medicalrouter = APIRouter()
logger = logging.getLogger("medical_audit")

# ----------------------- TOOL CONFIG (from spec) -----------------------

MEDICAL_NOTE_CONFIG: Dict[str, Any] = {
    "MedicalNoteAuditTool": {
        "Modules": [
            {
                "name": "Medical Note Audit",
                "workflow": [
                    "Upload Medical Note Documents (single or multiple progress notes)",
                    "Select Product",
                    "Select Standard",
                    "Select Requirements",
                    "Extract Contents of Document (PDF)",
                    "Process and Format Content",
                    "Pass to AI for Compliance Auditing",
                    "Generate Compliance / Non-Compliance Flags",
                    "Generate Recommendations",
                    "Score Compliance (0-100)",
                ],
            },
            {
                "name": "Face Sheet Extraction",
                "workflow": [
                    "Extract Data from Document",
                    "Parse Identifiers and Patient Demographics",
                    "Return Structured Output",
                ],
            },
        ],
        "Products": [
            {
                "id": "CTP",
                "name": "Cellular Tissue Product",
                "description": "Cellular & Tissue-based Products used for DFU/VLU wound treatment.",
                "standards": [
                    {
                        "id": "CTP-CMS-LCD-L39764",
                        "name": "CMS LCD L39764",
                        "source": "Centers for Medicare & Medicaid Services",
                        "description": "Coverage criteria for Skin Substitute Grafts/CTPs for DFUs & VLUs.",
                    },
                    {
                        "id": "CTP-CMS-LCD-L35041",
                        "name": "CMS LCD L35041",
                        "source": "Centers for Medicare & Medicaid Services",
                        "description": "Coverage determination for CTP use in DFUs & VLUs.",
                    },
                    {
                        "id": "CTP-CMS-BILLING",
                        "name": "CMS Billing & Coding Article (CTP)",
                        "source": "CMS",
                        "description": "Coding, HCPCS rules, documentation for wound application of CTPs.",
                    },
                    {
                        "id": "CTP-FDA-HCTP",
                        "name": "FDA HCT/P Guidance",
                        "source": "Food and Drug Administration",
                        "description": "Regulatory classification for human cellular & tissue-based products.",
                    },
                ],
            },
            {
                "id": "DME",
                "name": "Durable Medical Equipment",
                "description": "Durable Medical Equipment, Prosthetics, Orthotics & Supplies.",
                "standards": [
                    {
                        "id": "DME-CMS-ORDERREQ",
                        "name": "CMS DMEPOS Order Requirements",
                        "source": "CMS",
                        "description": "Documentation and order rules for DME suppliers.",
                    },
                    {
                        "id": "DME-OIG-GUIDANCE",
                        "name": "OIG Compliance Program Guidance (DMEPOS)",
                        "source": "Office of Inspector General",
                        "description": "Supplier-side compliance framework.",
                    },
                    {
                        "id": "DME-CMS-DOCREQ",
                        "name": "CMS Standard Documentation Requirements",
                        "source": "CMS/MAC",
                        "description": "Required documentation for all DMEPOS claims.",
                    },
                    {
                        "id": "DME-CODELISTS",
                        "name": "DME Procedure Code Lists & Coverage Manuals",
                        "source": "CMS / State Manuals",
                        "description": "HCPCS-based coverage and coding rules.",
                    },
                ],
            },
        ],
        "ComplianceCategories": [
            {
                "id": "WC",
                "name": "Wound Characteristics and Measurements",
                "description": "Wound size, depth, duration, and measurement documentation.",
            },
            {
                "id": "WCOND",
                "name": "Wound Condition",
                "description": "Status of wound bed, infection signs, drainage, appearance.",
            },
            {
                "id": "MN",
                "name": "Medical Necessity",
                "description": "Evidence supporting clinical need for product or service.",
            },
            {
                "id": "APPFUP",
                "name": "Application and Follow-Up",
                "description": "Proper application process, intervals, follow-up care, healing evaluation.",
            },
            {
                "id": "CODCOMP",
                "name": "Coding and Compliance",
                "description": "HCPCS/CPT coding, bundling rules, coverage rules, claim narrative elements.",
            },
        ],
    }
}

# ----------------------- DATA MODELS -----------------------


class FaceSheet(BaseModel):
    practice_name: Optional[str] = ""
    first_name: Optional[str] = ""
    middle_name: Optional[str] = ""
    last_name: Optional[str] = ""
    dob: Optional[str] = ""
    sex: Optional[str] = ""
    street_address: Optional[str] = ""
    unit_number: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    zip_code: Optional[str] = ""
    email: Optional[str] = ""
    primary_phone: Optional[str] = ""
    secondary_phone: Optional[str] = ""
    insurance_category: Optional[str] = ""
    insurance_name: Optional[str] = ""
    policy_number: Optional[str] = ""
    subscriber_name: Optional[str] = ""


class FaceSheetAuditResponse(BaseModel):
    facesheet: FaceSheet
    completeness_percentage: int
    missing_fields: List[str]


class WoundNoteAuditResponse(BaseModel):
    compliance_status: Dict[str, str]
    recommendations: Dict[str, List[str]]
    completeness_percentage: int


# Required fields for face sheet completeness
REQUIRED_FACE_FIELDS = {
    "first_name",
    "last_name",
    "dob",
    "sex",
    "street_address",
    "city",
    "state",
    "zip_code",
    "primary_phone",
    "insurance_name",
    "policy_number",
}

# ----------------------- PDF TEXT EXTRACTION -----------------------


def extract_pdf_text_local(file_bytes: bytes, *, page_limit: Optional[int] = None) -> str:
    """
    Local PDF text extraction. If the PDF is image-only (no embedded text),
    this returns an empty string.

    The `page_limit` allows us to cap work for very large files (useful for
    serverless environments where execution time is limited).
    """
    reader = PdfReader(BytesIO(file_bytes))
    pages_text: List[str] = []
    for idx, page in enumerate(reader.pages):
        if page_limit is not None and idx >= page_limit:
            break
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to extract text from page %s: %s", idx, exc)
            continue
    return "\n".join(pages_text).strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text using PyPDF2 and, if allowed, fall back to AWS Textract
    for image-only documents. Textract is optional and driven by environment
    variables so the service can run locally without AWS credentials.
    """
    text = extract_pdf_text_local(file_bytes, page_limit=settings.max_pdf_pages)

    if text.strip():
        return text

    if settings.textract_enabled:
        textract_text = textract_document_text(file_bytes, region=settings.aws_region)
        if textract_text:
            logger.info("Extracted text using Textract fallback")
        return textract_text

    logger.info("No embedded text found and Textract disabled; returning empty string")
    return text


# ----------------------- FACE SHEET PARSING + AUDIT -----------------------


def parse_facesheet_from_text(text: str) -> FaceSheet:
    """
    Parse a face sheet from a text-based PDF.
    Tuned loosely to your sample facesheets.
    """
    fs = FaceSheet()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Practice name: first non-empty line
    if lines:
        fs.practice_name = lines[0]

    # Patient name
    m = re.search(r"Patient['’]s Name:\s*([A-Za-z ,.'-]+)", text, re.IGNORECASE)
    if m:
        full = m.group(1).strip()
        parts = full.split()
        if len(parts) >= 2:
            fs.first_name = parts[0]
            fs.last_name = " ".join(parts[1:])
        else:
            fs.last_name = full

    # DOB
    m = re.search(r"(Date of Birth|DOB)[: ]+\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})", text, re.IGNORECASE)
    if m:
        fs.dob = m.group(2).strip()

    # Sex / gender
    m = re.search(r"Sex[: ]+\s*(Male|Female|M|F)\b", text, re.IGNORECASE)
    if m:
        fs.sex = m.group(1).title()

    # Street address – a rough heuristic
    for ln in lines:
        if "address" in ln.lower() and re.search(r"\d{1,5}\s+\w+", ln):
            parts = re.split(r"address[^:]*:", ln, flags=re.IGNORECASE)
            addr = parts[-1].strip() if len(parts) > 1 else ln.strip()
            fs.street_address = addr
            break

    # Primary phone
    m = re.search(r"(Primary Phone|Home Phone|Phone)[:#]?\s*([\d\-\(\) ]{7,})", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\(?\d{3}\)?[ \-]?\d{3}[ \-]?\d{4})", text)
    if m:
        fs.primary_phone = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else m.group(1).strip()

    # Zip code
    m = re.search(r"Zip Code[: ]*([0-9]{5})", text, re.IGNORECASE)
    if m:
        fs.zip_code = m.group(1)

    # Insurance name
    for ln in lines:
        if "primary insurance" in ln.lower():
            parts = ln.split(":")
            if len(parts) > 1:
                fs.insurance_name = parts[1].strip()
            break

    # Policy / subscriber ID
    m = re.search(r"(Sub ID|Subscriber ID|Policy Number)[:#]?\s*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
    if m:
        fs.policy_number = m.group(2).strip()

    return fs


def audit_facesheet(fs: FaceSheet) -> FaceSheetAuditResponse:
    data = fs.dict()
    missing = [f for f in REQUIRED_FACE_FIELDS if not data.get(f)]
    total = len(REQUIRED_FACE_FIELDS)
    completeness = int(((total - len(missing)) / total) * 100) if total else 0

    return FaceSheetAuditResponse(
        facesheet=fs,
        completeness_percentage=completeness,
        missing_fields=missing,
    )


# ----------------------- WOUND NOTE AUDIT -----------------------


def audit_wound_note(text: str) -> WoundNoteAuditResponse:
    """
    Rule-based wound note audit using ComplianceCategories from the spec
    (Wound Characteristics & Measurements, Wound Condition, Medical Necessity,
    Application & Follow-Up, Coding & Compliance).
    """
    lower = text.lower()

    compliance: Dict[str, str] = {}
    recs: Dict[str, List[str]] = {}

    # WC – Wound Characteristics and Measurements
    has_locations = "wound 1" in lower or "location:" in lower
    has_measurements = bool(re.search(r"\b\d+(\.\d+)?\s*cm\b", lower))
    has_stage = "stage:" in lower

    key = "Wound Characteristics and Measurements"
    if has_locations and has_measurements and has_stage:
        compliance[key] = "Compliant"
        recs[key] = []
    else:
        compliance[key] = "Non-Compliant"
        recs[key] = [
            "Include wound locations, stages, and length/width/depth measurements for each wound.",
            "Document drainage amount, odor, and tissue composition (e.g., % slough, % granulation).",
        ]

    # WCOND – Wound Condition
    has_drainage = "drainage" in lower
    has_tissue = "tissue" in lower or "slough" in lower or "granulation" in lower
    has_periwound = "periwound" in lower or "periwound skin" in lower or "periwound condition" in lower

    key = "Wound Condition"
    if has_drainage and has_tissue and has_periwound:
        compliance[key] = "Compliant"
        recs[key] = []
    else:
        compliance[key] = "Non-Compliant"
        recs[key] = [
            "Describe wound base and periwound condition (e.g., erythema, maceration, excoriation).",
            "Include tissue type and percentage (e.g., 60% slough, 40% granulation).",
        ]

    # MN – Medical Necessity
    has_history = "history of wound" in lower or "transition of care" in lower or "hpi" in lower
    has_prior_tx = "treatments tried" in lower or "current treatment" in lower or "has been using" in lower
    has_pain = "pain level" in lower or "in a lot of pain" in lower
    has_diagnostics = "culture" in lower or "xray" in lower or "imaging" in lower
    has_allograft_rationale = "allograft" in lower or "move forward with allografts" in lower

    key = "Medical Necessity"
    if has_history and has_prior_tx and has_pain and (has_diagnostics or has_allograft_rationale):
        compliance[key] = "Compliant"
        recs[key] = []
    else:
        compliance[key] = "Non-Compliant"
        recs[key] = [
            "Describe the wound history and conservative therapy tried/failed.",
            "Document diagnostics (cultures, imaging) and explain the rationale for advanced therapy (e.g., allografts).",
        ]

    # APPFUP – Application and Follow-Up
    has_plan = "plan:" in lower or "plan of care" in lower
    has_dressing = "cover with" in lower or ("apply" in lower and "dressing" in lower)
    has_frequency = "x weekly" in lower or "2 x weekly" in lower or "3 x weekly" in lower
    has_followup = "follow up" in lower

    key = "Application and Follow-Up"
    if has_plan and has_dressing and has_frequency and has_followup:
        compliance[key] = "Compliant"
        recs[key] = []
    else:
        compliance[key] = "Non-Compliant"
        recs[key] = [
            "Document procedure details, including products used and layer sequence.",
            "Include dressing change frequency, who will perform changes, and follow-up timeframe.",
        ]

    # CODCOMP – Coding and Compliance
    has_icd = bool(re.search(r"\b[lL]\d{2}\.\d+", text))  # rough ICD-10 pattern
    has_cpt = "procedure codes" in lower or "97597" in lower or "97598" in lower

    key = "Coding and Compliance"
    if has_icd and has_cpt:
        compliance[key] = "Compliant"
        recs[key] = []
    else:
        compliance[key] = "Non-Compliant"
        recs[key] = [
            "Link ICD-10 codes to documented wounds.",
            "Include wound-care CPT codes with units and documentation supporting each.",
        ]

    total = len(compliance)
    compliant_count = sum(1 for v in compliance.values() if v == "Compliant")
    completeness_pct = int(round((compliant_count / total) * 100)) if total else 0

    return WoundNoteAuditResponse(
        compliance_status=compliance,
        recommendations=recs,
        completeness_percentage=completeness_pct,
    )


# ----------------------- DOCUMENT TYPE DETECTION -----------------------


def detect_document_type(file_bytes: bytes) -> str:
    """
    Heuristic detection: 'facesheet' vs 'wound_note' vs 'image_only'.
    """
    text = extract_pdf_text(file_bytes)
    lower = text.lower()

    if not lower.strip():
        return "image_only"

    face_sheet_keywords = [
        "patient registration form",
        "patient enrollment",
        "primary insurance info",
        "emergency contact info",
        "account #",
        "wound care hawai",
        "many plans require prior authorization",
        "health insurance information",
    ]

    wound_note_keywords = [
        "office visit note",
        "chief complaint",
        "hpi:",
        "history of present illness",
        "assessment/plan",
        "procedure codes",
        "wound 1:",
        "wound 2:",
        "wound 3:",
    ]

    if any(k in lower for k in face_sheet_keywords):
        return "facesheet"

    if any(k in lower for k in wound_note_keywords):
        return "wound_note"

    if "office visit" in lower or "exam" in lower:
        return "wound_note"

    return "facesheet"


# ----------------------- ROUTES -----------------------


@medicalrouter.get("/config")
async def get_medical_note_config() -> Dict[str, Any]:
    """
    Expose the tool configuration (modules, products, standards, categories).
    Matches the JSON spec you uploaded.
    """
    return MEDICAL_NOTE_CONFIG


@medicalrouter.post("/audit/face-sheet", response_model=FaceSheetAuditResponse)
async def audit_face_sheet(file: UploadFile = File(...)):
    """
    Explicit face-sheet audit endpoint (for text-based PDFs).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads allowed.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF uploaded.")

    text = extract_pdf_text(file_bytes)
    if not text.strip():
        detail = (
            "No text could be extracted. Enable MED_AUDIT_TEXTRACT_ENABLED for OCR "
            "or provide a text-based PDF."
        )
        raise HTTPException(status_code=400, detail=detail)

    facesheet = parse_facesheet_from_text(text)
    return audit_facesheet(facesheet)


@medicalrouter.post("/audit/wound-note", response_model=WoundNoteAuditResponse)
async def audit_wound_note_endpoint(file: UploadFile = File(...)):
    """
    Explicit wound-note audit endpoint.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads allowed.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF uploaded.")

    text = extract_pdf_text(file_bytes)
    if not text.strip():
        detail = (
            "No text could be extracted. Enable MED_AUDIT_TEXTRACT_ENABLED for OCR "
            "or provide a text-based PDF."
        )
        raise HTTPException(status_code=400, detail=detail)

    return audit_wound_note(text)


@medicalrouter.post("/audit/auto")
async def audit_auto(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload ANY medical PDF.
    Auto-detect type and run the appropriate audit.
    Response is shaped to align with the spec:
    - module name
    - product / standards (defaults for now)
    - compliance categories + score
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads allowed.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF uploaded.")

    doc_type = detect_document_type(file_bytes)

    if doc_type == "image_only":
        raise HTTPException(
            status_code=400,
            detail=(
                "PDF appears to be image-only (no embedded text). "
                "Enable MED_AUDIT_TEXTRACT_ENABLED to OCR and audit these files."
            ),
        )

    # Defaults for now: CTP / LCD L39764 (can be parameterized later)
    product = MEDICAL_NOTE_CONFIG["MedicalNoteAuditTool"]["Products"][0]
    standards = product["standards"]

    if doc_type == "facesheet":
        text = extract_pdf_text(file_bytes)
        facesheet = parse_facesheet_from_text(text)
        audit = audit_facesheet(facesheet)
        return {
            "module": "Face Sheet Extraction",
            "document_type": "facesheet",
            "product": None,
            "standards": [],
            "audit": audit.dict(),
            "textract_enabled": settings.textract_enabled,
        }

    # wound note
    text = extract_pdf_text(file_bytes)
    audit = audit_wound_note(text)
    return {
        "module": "Medical Note Audit",
        "document_type": "wound_note",
        "product": {"id": product["id"], "name": product["name"]},
        "standards": [{"id": s["id"], "name": s["name"]} for s in standards],
        "audit": audit.dict(),
        "score": audit.completeness_percentage,
        "textract_enabled": settings.textract_enabled,
    }


@medicalrouter.get("/upload", response_class=HTMLResponse)
async def medical_upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Document Audit</title>
        <meta charset="utf-8" />
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }
            h1 { margin-bottom: 0.5rem; }
            .card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; }
            .result { white-space: pre-wrap; background: #f7f7f7; padding: 12px; margin-top: 16px; border-radius: 4px; }
            .error { color: #b00020; margin-top: 8px; }
            button { padding: 8px 16px; cursor: pointer; }
            input[type="file"] { margin-top: 8px; margin-bottom: 12px; }
        </style>
    </head>
    <body>
        <h1>Medical Document Audit</h1>
        <p>Upload a PDF face sheet or wound note and get an audit.</p>

        <div class="card">
            <input id="file-input" type="file" accept="application/pdf" />
            <br/>
            <button onclick="upload()">Upload &amp; Audit</button>
            <div id="status"></div>
            <div id="error" class="error"></div>
            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
            async function upload() {
                const fileInput = document.getElementById('file-input');
                const statusEl = document.getElementById('status');
                const errorEl = document.getElementById('error');
                const resultEl = document.getElementById('result');
                errorEl.textContent = '';
                resultEl.style.display = 'none';
                resultEl.textContent = '';

                if (!fileInput.files.length) {
                    errorEl.textContent = 'Please choose a PDF file first.';
                    return;
                }

                const file = fileInput.files[0];
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    errorEl.textContent = 'Only PDF files are allowed.';
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                statusEl.textContent = 'Uploading and processing...';

                try {
                    const resp = await fetch('/medical/audit/auto', {
                        method: 'POST',
                        body: formData
                    });

                    if (!resp.ok) {
                        const text = await resp.text();
                        throw new Error(text || ('HTTP ' + resp.status));
                    }

                    const json = await resp.json();
                    statusEl.textContent = 'Audit complete. Detected type: ' + json.document_type;
                    resultEl.style.display = 'block';
                    resultEl.textContent = JSON.stringify(json, null, 2);
                } catch (err) {
                    console.error(err);
                    statusEl.textContent = '';
                    errorEl.textContent = 'Error: ' + err.message;
                }
            }
        </script>
    </body>
    </html>
    """
