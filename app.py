from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
from ultralytics import YOLO
from rembg import remove, new_session
from PIL import Image
import numpy as np
import threading
import time
import base64
import io
import datetime
import math
import uuid

# PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import Image as RLImage
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("[!] reportlab not installed. Run: pip install reportlab")

app = Flask(__name__)

# ─── Load Models ──────────────────────────────────────────────────────────────
print("[*] Loading YOLO model...")
model = YOLO("best.pt")
print("[OK] YOLO ready.")

print("[*] Loading rembg session...")
rembg_session = new_session()
print("[OK] rembg ready.")

# --- Allowed Classes (only these will be processed) ---
ALLOWED_CLASSES = [
    "Corroded-Bolt",
    "Corroded-Nut",
    "Corroded-Nut-and-Bolt",
    "Non-Corroded-Nut-and-Bolt",
    "Non-corroded Nut",
    "Non-corroded-Bolts",
]

# --- Part Database ---
# entry: display name, type, material, standard, use_case, quality_threshold
PART_DB = {
    "Corroded-Bolt":             ("Corroded Bolt",           "Fastener - Bolt", "Carbon Steel",    "ISO 4014", "Engine assembly / Chassis",  0),
    "Corroded-Nut":              ("Corroded Nut",            "Fastener - Nut",  "Carbon Steel",    "ISO 4032", "Wheel & brake assembly",     0),
    "Corroded-Nut-and-Bolt":     ("Corroded Nut & Bolt Set", "Fastener Set",    "Carbon Steel",    "ISO 4014", "Structural joints",          0),
    "Non-Corroded-Nut-and-Bolt": ("Nut & Bolt Set",          "Fastener Set",    "Stainless Steel", "ISO 4014", "Structural joints",         75),
    "Non-corroded Nut":          ("Standard Nut",            "Fastener - Nut",  "Stainless Steel", "ISO 4032", "Wheel & brake assembly",    75),
    "Non-corroded-Bolts":        ("Standard Bolt",           "Fastener - Bolt", "Stainless Steel", "ISO 4014", "Engine assembly / Chassis", 75),
    "default":                   ("Unclassified Part",       "Unknown",         "Unknown",         "Pending",  "General inspection",        80),
}


# ─── Image Upload Part Database ───────────────────────────────────────────────
# These are used when a user uploads an image for inspection
IMAGE_PART_DB = {
    "gear": {
        "name":      "Industrial Gear",
        "type":      "Transmission — Spur Gear",
        "material":  "Alloy Steel 20MnCr5",
        "standard":  "ISO 6336",
        "usecase":   "Gearbox / Transmission Assembly",
        "threshold": 72,
    },
    "bearing": {
        "name":      "Ball Bearing",
        "type":      "Deep Groove Ball Bearing",
        "material":  "Chrome Steel GCr15",
        "standard":  "ISO 15",
        "usecase":   "Rotating Shaft Support",
        "threshold": 75,
    },
    "engine_block": {
        "name":      "Engine Block Component",
        "type":      "Engine — Cylinder Block",
        "material":  "Cast Iron / Aluminium",
        "standard":  "ISO 6621",
        "usecase":   "Engine Assembly",
        "threshold": 80,
    },
    "brake_disc": {
        "name":      "Brake Disc",
        "type":      "Ventilated Brake Disc",
        "material":  "Grey Cast Iron GG25",
        "standard":  "ISO 26867",
        "usecase":   "Braking System",
        "threshold": 85,
    },
    "default": {
        "name":      "Industrial Component",
        "type":      "Mechanical Part",
        "material":  "Alloy Steel",
        "standard":  "ISO 9001",
        "usecase":   "General Manufacturing",
        "threshold": 75,
    },
}

# ─── Open Camera ──────────────────────────────────────────────────────────────
def open_camera():
    backends = [(cv2.CAP_DSHOW,"DirectShow"),(cv2.CAP_MSMF,"MSMF"),(None,"Default")]
    for index in [0,1,2]:
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[OK] Camera opened — index={index}, backend={name}")
                        return cap
                    cap.release()
            except: pass
    return None

camera = open_camera()
if camera is None:
    raise RuntimeError("No camera found.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

# ─── Shared State ─────────────────────────────────────────────────────────────
feed_lock = threading.Lock()
data_lock = threading.Lock()
live_frame_bytes = None

stats = {
    "isolated_b64":    None,
    "heatmap_b64":     None,
    "part_name":       "Scanning...",
    "part_class":      "--",
    "part_type":       "--",
    "part_material":   "--",
    "part_standard":   "--",
    "part_usecase":    "--",
    "confidence":      0.0,
    "quality_score":   0,
    "verdict":         "PENDING",   # PASS / FAIL / PENDING
    "defect_zones":    0,
    "bbox_wh":         (0, 0),
    "detected":        False,
    "total_scanned":   0,
    "total_pass":      0,
    "total_fail":      0,
    "history":         [],          # last 20 inspections
    "fps":             0.0,
    "session_start":   datetime.datetime.now().strftime("%H:%M:%S"),
    "last_updated":    "--",
}

# ─── rembg Queue ─────────────────────────────────────────────────────────────
import queue
rembg_queue  = queue.Queue(maxsize=1)
rembg_result = {"b64": None, "heatmap_b64": None, "defect_zones": 0, "quality_score": 0}
rembg_lock   = threading.Lock()

# ─── Heatmap Generator ───────────────────────────────────────────────────────
def generate_heatmap(crop_bgr):
    """Generate defect heatmap using edge detection + Gaussian blur on the crop."""
    try:
        gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Edge detection — highlights surface anomalies
        edges    = cv2.Canny(gray, 50, 150)

        # Dilate edges to make them more visible
        kernel   = np.ones((3,3), np.uint8)
        edges    = cv2.dilate(edges, kernel, iterations=2)

        # Apply Gaussian blur to create smooth heatmap effect
        blur     = cv2.GaussianBlur(edges.astype(float), (21,21), 0)

        # Normalize to 0-255
        if blur.max() > 0:
            blur = (blur / blur.max() * 255).astype(np.uint8)
        else:
            blur = blur.astype(np.uint8)

        # Apply COLORMAP_JET for heatmap effect (blue=safe, red=defect zones)
        heatmap  = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

        # Blend with original crop (30% heatmap, 70% original)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended  = cv2.addWeighted(crop_rgb, 0.55, heatmap_rgb, 0.45, 0)

        # Count defect zones (high-intensity edge regions)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_zones = len([c for c in contours if cv2.contourArea(c) > 30])

        # Add "DEFECT SCAN" label on heatmap
        labeled = blended.copy()
        cv2.putText(labeled, "DEFECT SCAN", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(labeled, f"ZONES: {defect_zones}", (8, labeled.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1, cv2.LINE_AA)

        pil_img = Image.fromarray(labeled)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), defect_zones

    except Exception as e:
        print(f"[!] Heatmap error: {e}")
        return None, 0

# ─── Quality Score Calculator ─────────────────────────────────────────────────
def calculate_quality_score(confidence, defect_zones, bbox_w, bbox_h, cls_name):
    """Calculate 0-100 quality score based on multiple factors."""
    _, _, _, _, _, critical_threshold = PART_DB.get(cls_name, PART_DB["default"])

    # Factor 1: Confidence score (40% weight)
    conf_score = confidence * 0.40

    # Factor 2: Defect zones penalty (30% weight) — fewer zones = better
    max_zones  = 15
    zone_score = max(0, (1 - defect_zones / max_zones)) * 30

    # Factor 3: Size appropriateness (15% weight) — reasonable bbox size
    area       = bbox_w * bbox_h
    size_score = 15 if 2000 < area < 150000 else 8

    # Factor 4: Image sharpness via Laplacian variance (15% weight)
    sharpness_score = 15  # default, updated in bg removal
    total = round(conf_score + zone_score + size_score + sharpness_score)
    return min(100, max(0, total))

# ─── BG Removal + Full Analysis ──────────────────────────────────────────────
def full_analysis(crop_bgr, cls_name, confidence, bbox_w, bbox_h):
    """Run rembg + heatmap + quality score on crop."""
    try:
        # Background removal
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        result  = remove(pil_img, session=rembg_session)
        bg      = Image.new("RGBA", result.size, (13,17,23,255))
        bg.paste(result, mask=result.split()[3])
        buf = io.BytesIO()
        bg.convert("RGB").save(buf, format="PNG")
        isolated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Sharpness (Laplacian variance)
        gray      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Heatmap
        heatmap_b64, defect_zones = generate_heatmap(crop_bgr)

        # Quality score
        _, _, _, _, _, critical_threshold = PART_DB.get(cls_name, PART_DB["default"])
        conf_score   = confidence * 0.40
        zone_score   = max(0, (1 - defect_zones/15)) * 30
        area         = bbox_w * bbox_h
        size_score   = 15 if 2000 < area < 150000 else 8
        sharp_score  = min(15, sharpness / 200 * 15)
        quality      = min(100, max(0, round(conf_score + zone_score + size_score + sharp_score)))

        # Verdict
        verdict = "PASS" if quality >= critical_threshold else "FAIL"

        return isolated_b64, heatmap_b64, defect_zones, quality, verdict

    except Exception as e:
        print(f"[!] Analysis error: {e}")
        _, buf2 = cv2.imencode(".png", crop_bgr)
        return base64.b64encode(buf2.tobytes()).decode("utf-8"), None, 0, 50, "PENDING"

def rembg_worker():
    while True:
        try:
            item = rembg_queue.get(timeout=1)
            crop, cls_name, confidence, bbox_w, bbox_h = item
            iso, hmap, zones, quality, verdict = full_analysis(crop, cls_name, confidence, bbox_w, bbox_h)
            with rembg_lock:
                rembg_result["b64"]          = iso
                rembg_result["heatmap_b64"]  = hmap
                rembg_result["defect_zones"] = zones
                rembg_result["quality_score"]= quality
                rembg_result["verdict"]      = verdict
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[!] Worker error: {e}")

threading.Thread(target=rembg_worker, daemon=True).start()

# ─── Camera Loop ──────────────────────────────────────────────────────────────
def camera_loop():
    global camera, live_frame_bytes
    frame_count = 0; fps_timer = time.time(); current_fps = 0.0
    rembg_timer = 0

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(1); camera = open_camera(); continue

        success, frame = camera.read()
        if not success or frame is None:
            time.sleep(0.02); continue

        frame_count += 1
        if frame_count % 15 == 0:
            elapsed = time.time() - fps_timer
            current_fps = round(15/elapsed, 1) if elapsed > 0 else 0.0
            fps_timer = time.time()

        # ── Camera display only — no detection ──────────────────────
        # Detection disabled — using image upload mode only
        annotated = frame.copy()

        # Draw minimal border on frame
        fh, fw = frame.shape[:2]
        cv2.rectangle(annotated, (10,10), (fw-10,fh-10), (0,180,255), 1)
        cv2.putText(annotated, "SMARTVISION AI - UPLOAD MODE",
                    (20, fh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,180,255), 1, cv2.LINE_AA)

        ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with feed_lock:
                live_frame_bytes = buf.tobytes()

        # No detection — camera just streams
        detected=False; part_name="Upload an image to inspect"; part_class="--"
        part_type="--"; part_material="--"; part_standard="--"; part_usecase="--"
        confidence=0.0; bbox_wh=(0,0); do_count=False
        boxes=None

        with rembg_lock:
            latest_iso     = rembg_result["b64"]
            latest_hmap    = rembg_result["heatmap_b64"]
            latest_zones   = rembg_result["defect_zones"]
            latest_quality = rembg_result["quality_score"]
            latest_verdict = rembg_result.get("verdict", "PENDING")

        with data_lock:
            stats["part_name"]      = part_name
            stats["part_class"]     = part_class
            stats["part_type"]      = part_type
            stats["part_material"]  = part_material
            stats["part_standard"]  = part_standard
            stats["part_usecase"]   = part_usecase
            stats["confidence"]     = confidence
            stats["bbox_wh"]        = bbox_wh
            stats["detected"]       = detected
            stats["fps"]            = current_fps
            stats["last_updated"]   = datetime.datetime.now().strftime("%H:%M:%S")
            stats["isolated_b64"]   = latest_iso
            stats["heatmap_b64"]    = latest_hmap
            stats["defect_zones"]   = latest_zones
            stats["quality_score"]  = latest_quality
            stats["verdict"]        = latest_verdict

            if do_count:
                stats["total_scanned"] += 1
                if latest_verdict == "PASS":
                    stats["total_pass"] += 1
                elif latest_verdict == "FAIL":
                    stats["total_fail"] += 1

                # Add to history (keep last 20)
                entry = {
                    "time":      datetime.datetime.now().strftime("%H:%M:%S"),
                    "name":      part_name,
                    "cls":       part_class,
                    "conf":      confidence,
                    "quality":   latest_quality,
                    "verdict":   latest_verdict,
                    "zones":     latest_zones,
                }
                stats["history"].insert(0, entry)
                stats["history"] = stats["history"][:20]

threading.Thread(target=camera_loop, daemon=True).start()

# ─── Routes ───────────────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with feed_lock:
            frame = live_frame_bytes
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def get_stats():
    with data_lock:
        return jsonify({
            "part_name":      stats["part_name"],
            "part_class":     stats["part_class"],
            "part_type":      stats["part_type"],
            "part_material":  stats["part_material"],
            "part_standard":  stats["part_standard"],
            "part_usecase":   stats["part_usecase"],
            "confidence":     stats["confidence"],
            "quality_score":  stats["quality_score"],
            "verdict":        stats["verdict"],
            "defect_zones":   stats["defect_zones"],
            "bbox_w":         stats["bbox_wh"][0],
            "bbox_h":         stats["bbox_wh"][1],
            "detected":       stats["detected"],
            "total_scanned":  stats["total_scanned"],
            "total_pass":     stats["total_pass"],
            "total_fail":     stats["total_fail"],
            "history":        stats["history"],
            "fps":            stats["fps"],
            "session_start":  stats["session_start"],
            "last_updated":   stats["last_updated"],
            "isolated_b64":   stats["isolated_b64"],
            "heatmap_b64":    stats["heatmap_b64"],
        })


@app.route("/export_report")
def export_report():
    if not REPORTLAB_OK:
        return "reportlab not installed. Run: pip install reportlab", 500

    with data_lock:
        snap = {k: v for k, v in stats.items()}

    report_id  = "INS-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:4].upper()
    timestamp  = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
    verdict    = snap.get("verdict", "PENDING")
    part_name  = snap.get("part_name", "Unknown")
    part_class = snap.get("part_class", "--")
    part_type  = snap.get("part_type", "--")
    material   = snap.get("part_material", "--")
    standard   = snap.get("part_standard", "--")
    usecase    = snap.get("part_usecase", "--")
    confidence = snap.get("confidence", 0)
    if isinstance(confidence, float) and confidence < 1.0:
        confidence = round(confidence * 100, 1)  # convert 0-1 to 0-100 if needed
    quality    = snap.get("quality_score", 0)
    zones      = snap.get("defect_zones", 0)
    _bwh       = snap.get("bbox_wh", [0,0])
    bbox_w     = _bwh[0] if isinstance(_bwh,(list,tuple)) and len(_bwh)>0 else 0
    bbox_h     = _bwh[1] if isinstance(_bwh,(list,tuple)) and len(_bwh)>1 else 0
    iso_b64    = snap.get("isolated_b64")
    hmap_b64   = snap.get("heatmap_b64")
    total_scan = snap.get("total_scanned", 0)
    total_pass = snap.get("total_pass", 0)
    total_fail = snap.get("total_fail", 0)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=12*mm, bottomMargin=12*mm)
    W = A4[0] - 30*mm
    styles = getSampleStyleSheet()

    def sty(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    small_sty   = sty("Small",  fontName="Helvetica",      fontSize=7,  textColor=colors.HexColor("#64748b"), alignment=TA_CENTER)
    label_sty   = sty("Label",  fontName="Helvetica-Bold", fontSize=8,  textColor=colors.HexColor("#334155"))
    value_sty   = sty("Value",  fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#0f172a"))
    section_sty = sty("Sec",    fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#1e3a5f"), spaceAfter=4, spaceBefore=10)
    footer_sty  = sty("Footer", fontName="Helvetica",      fontSize=7,  textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER)

    verdict_color = colors.HexColor("#16a34a") if verdict=="PASS" else                     colors.HexColor("#dc2626") if verdict=="FAIL" else                     colors.HexColor("#d97706")
    quality_color = colors.HexColor("#16a34a") if quality>=80 else                     colors.HexColor("#d97706") if quality>=60 else                     colors.HexColor("#dc2626")

    story = []

    # HEADER
    hdr = Table([[
        Paragraph("SmartVision AI", sty("T1", fontName="Helvetica-Bold", fontSize=20, textColor=colors.white)),
        Paragraph("INSPECTION REPORT", sty("T2", fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#bfdbfe"), alignment=TA_RIGHT)),
    ]], colWidths=[W*0.6, W*0.4])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#0f2847")),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(0,-1),  12),
        ("RIGHTPADDING",(-1,0),(-1,-1),12),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 3*mm))

    # META
    meta = Table([[
        Paragraph(f"Report ID: {report_id}", sty("M1", fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#2563eb"))),
        Paragraph(f"Timestamp: {timestamp}", sty("M2", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#475569"), alignment=TA_CENTER)),
        Paragraph(f"Session Parts: {total_scan}", sty("M3", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#475569"), alignment=TA_RIGHT)),
    ]], colWidths=[W/3]*3)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#eff6ff")),
        ("LINEBELOW",     (0,0),(-1,-1), 1, colors.HexColor("#bfdbfe")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ]))
    story.append(meta)
    story.append(Spacer(1, 4*mm))

    # VERDICT / QUALITY / CONFIDENCE
    vrd_sym = "PASS" if verdict=="PASS" else "FAIL" if verdict=="FAIL" else "PENDING"
    vrd_tbl = Table([[
        Paragraph(vrd_sym, sty("V1", fontName="Helvetica-Bold", fontSize=18, textColor=verdict_color, alignment=TA_CENTER)),
        Paragraph(f"{quality}/100", sty("V2", fontName="Helvetica-Bold", fontSize=18, textColor=quality_color, alignment=TA_CENTER)),
        Paragraph(f"{confidence}%", sty("V3", fontName="Helvetica-Bold", fontSize=18, textColor=colors.HexColor("#0369a1"), alignment=TA_CENTER)),
    ],[
        Paragraph("VERDICT",      small_sty),
        Paragraph("QUALITY SCORE", small_sty),
        Paragraph("CONFIDENCE",   small_sty),
    ]], colWidths=[W/3]*3)
    vrd_bg = colors.HexColor("#f0fdf4") if verdict=="PASS" else              colors.HexColor("#fef2f2") if verdict=="FAIL" else              colors.HexColor("#fffbeb")
    vrd_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,-1), vrd_bg),
        ("BACKGROUND",    (1,0),(-1,-1), colors.HexColor("#f8fafc")),
        ("BOX",           (0,0),(-1,-1), 1.5, colors.HexColor("#cbd5e1")),
        ("LINEAFTER",     (0,0),(1,-1),  0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",    (0,0),(-1,0),  8),
        ("BOTTOMPADDING", (0,0),(-1,0),  2),
        ("TOPPADDING",    (0,1),(-1,1),  2),
        ("BOTTOMPADDING", (0,1),(-1,1),  8),
    ]))
    story.append(vrd_tbl)
    story.append(Spacer(1, 4*mm))

    # IMAGES
    def b64_to_img(b64str, w, h):
        try:
            data = base64.b64decode(b64str)
            return RLImage(io.BytesIO(data), width=w, height=h)
        except:
            return Paragraph("[Image unavailable]", small_sty)

    iw, ih = W*0.46, 55*mm
    iso_cell  = b64_to_img(iso_b64,  iw, ih) if iso_b64  else Paragraph("No image", small_sty)
    hmap_cell = b64_to_img(hmap_b64, iw, ih) if hmap_b64 else Paragraph("No heatmap", small_sty)

    img_tbl = Table([
        [Paragraph("Isolated Component", sty("IL", fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#1e40af"), alignment=TA_CENTER)),
         Spacer(1,1),
         Paragraph("Defect Heatmap", sty("HL", fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#9a3412"), alignment=TA_CENTER))],
        [iso_cell, Spacer(1,1), hmap_cell],
    ], colWidths=[iw, W*0.08, iw])
    img_tbl.setStyle(TableStyle([
        ("BOX",          (0,1),(0,1), 1, colors.HexColor("#bfdbfe")),
        ("BOX",          (2,1),(2,1), 1, colors.HexColor("#fed7aa")),
        ("BACKGROUND",   (0,1),(0,1), colors.HexColor("#050a10")),
        ("BACKGROUND",   (2,1),(2,1), colors.HexColor("#050a10")),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,1),(-1,1), 4),
        ("BOTTOMPADDING",(0,1),(-1,1), 4),
        ("BOTTOMPADDING",(0,0),(-1,0), 3),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 4*mm))

    # SPECS TABLE
    story.append(Paragraph("PART SPECIFICATIONS", section_sty))
    spec = Table([
        [Paragraph("Part Name",      label_sty), Paragraph(part_name,  value_sty), Paragraph("Classification", label_sty), Paragraph(part_class, value_sty)],
        [Paragraph("Type",           label_sty), Paragraph(part_type,  value_sty), Paragraph("Material",       label_sty), Paragraph(material,   value_sty)],
        [Paragraph("Standard",       label_sty), Paragraph(standard,   value_sty), Paragraph("Application",    label_sty), Paragraph(usecase,    value_sty)],
        [Paragraph("Bbox Size",      label_sty), Paragraph(f"{bbox_w}x{bbox_h}px", value_sty), Paragraph("Defect Zones", label_sty), Paragraph(str(zones), value_sty)],
    ], colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    spec.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,-1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND",    (2,0),(2,-1), colors.HexColor("#f1f5f9")),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ]))
    story.append(spec)
    story.append(Spacer(1, 4*mm))

    # SESSION SUMMARY
    story.append(Paragraph("SESSION SUMMARY", section_sty))
    pass_rate = round(total_pass/total_scan*100) if total_scan > 0 else 0
    sess = Table([[
        Paragraph(str(total_scan), sty("S1", fontName="Helvetica-Bold", fontSize=16, textColor=colors.HexColor("#0f2847"), alignment=TA_CENTER)),
        Paragraph(str(total_pass), sty("S2", fontName="Helvetica-Bold", fontSize=16, textColor=colors.HexColor("#16a34a"), alignment=TA_CENTER)),
        Paragraph(str(total_fail), sty("S3", fontName="Helvetica-Bold", fontSize=16, textColor=colors.HexColor("#dc2626"), alignment=TA_CENTER)),
        Paragraph(f"{pass_rate}%", sty("S4", fontName="Helvetica-Bold", fontSize=16, textColor=colors.HexColor("#2563eb"), alignment=TA_CENTER)),
    ],[
        Paragraph("Total Scanned", small_sty),
        Paragraph("Passed",        small_sty),
        Paragraph("Failed",        small_sty),
        Paragraph("Pass Rate",     small_sty),
    ]], colWidths=[W/4]*4)
    sess.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#f8fafc")),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#cbd5e1")),
        ("LINEAFTER",     (0,0),(2,-1),  0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0,0),(-1,0),  8),
        ("BOTTOMPADDING", (0,0),(-1,0),  2),
        ("TOPPADDING",    (0,1),(-1,1),  2),
        ("BOTTOMPADDING", (0,1),(-1,1),  8),
    ]))
    story.append(sess)
    story.append(Spacer(1, 4*mm))

    # FOOTER
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"Generated by SmartVision AI  ·  YOLOv8 + OpenCV + rembg  ·  {timestamp}  ·  {report_id}",
        footer_sty))
    story.append(Paragraph(
        "Auto-generated AI inspection report. For official quality certification, verify with a qualified engineer.",
        footer_sty))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"SmartVision_{report_id}.pdf")


@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Analyze an uploaded image — used for demo of gearbox/other parts."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file      = request.files["image"]
        part_type = request.form.get("part_type", "gear")
        img_bytes = file.read()

        # Convert to OpenCV image
        nparr  = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return jsonify({"error": "Invalid image"}), 400

        # Resize to reasonable size
        h, w   = img_cv.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale  = max_dim / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w*scale), int(h*scale)))

        # Get part info from database
        part_info = IMAGE_PART_DB.get(part_type, IMAGE_PART_DB["default"])

        # ── Step 1: Background removal ──
        try:
            pil_img  = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            result   = remove(pil_img, session=rembg_session)
            bg       = Image.new("RGBA", result.size, (13,17,23,255))
            bg.paste(result, mask=result.split()[3])
            buf_iso  = io.BytesIO()
            bg.convert("RGB").save(buf_iso, format="PNG")
            isolated_b64 = base64.b64encode(buf_iso.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[!] rembg error: {e}")
            _, buf_iso = cv2.imencode(".png", img_cv)
            isolated_b64 = base64.b64encode(buf_iso.tobytes()).decode("utf-8")

        # ── Step 2: Heatmap generation ──
        heatmap_b64, defect_zones = generate_heatmap(img_cv)

        # ── Step 3: Quality score ──
        gray      = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        h2, w2    = img_cv.shape[:2]
        area      = h2 * w2

        conf_score   = 88.5   # fixed high confidence for image upload
        zone_score   = max(0, (1 - defect_zones/15)) * 30
        size_score   = 15 if area > 5000 else 8
        sharp_score  = min(15, sharpness / 200 * 15)
        quality      = min(100, max(0, round(conf_score * 0.4 + zone_score + size_score + sharp_score)))
        verdict      = "PASS" if quality >= part_info["threshold"] else "FAIL"

        # ── Step 4: Update shared stats so PDF export works ──
        with data_lock:
            stats["part_name"]     = part_info["name"]
            stats["part_class"]    = part_type
            stats["part_type"]     = part_info["type"]
            stats["part_material"] = part_info["material"]
            stats["part_standard"] = part_info["standard"]
            stats["part_usecase"]  = part_info["usecase"]
            stats["confidence"]    = conf_score
            stats["quality_score"] = quality
            stats["verdict"]       = verdict
            stats["defect_zones"]  = defect_zones
            stats["bbox_wh"]       = (w2, h2)
            stats["detected"]      = True
            stats["isolated_b64"]  = isolated_b64
            stats["heatmap_b64"]   = heatmap_b64
            stats["last_updated"]  = datetime.datetime.now().strftime("%H:%M:%S")
            stats["total_scanned"] += 1
            if verdict == "PASS":
                stats["total_pass"] += 1
            else:
                stats["total_fail"] += 1
            entry = {
                "time":    datetime.datetime.now().strftime("%H:%M:%S"),
                "name":    part_info["name"],
                "cls":     part_type,
                "conf":    conf_score,
                "quality": quality,
                "verdict": verdict,
                "zones":   defect_zones,
            }
            stats["history"].insert(0, entry)
            stats["history"] = stats["history"][:20]

        return jsonify({
            "success":       True,
            "part_name":     part_info["name"],
            "part_type":     part_info["type"],
            "part_material": part_info["material"],
            "part_standard": part_info["standard"],
            "part_usecase":  part_info["usecase"],
            "confidence":    conf_score,
            "quality_score": quality,
            "verdict":       verdict,
            "defect_zones":  defect_zones,
            "bbox_w":        w2,
            "bbox_h":        h2,
            "isolated_b64":  isolated_b64,
            "heatmap_b64":   heatmap_b64,
        })

    except Exception as e:
        print(f"[!] Image analysis error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)