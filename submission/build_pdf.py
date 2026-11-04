"""Build the single submission PDF: links, BITS-Lab usage screenshots, and README."""
import json, os, re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether
from reportlab.lib.utils import ImageReader
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
cfg = json.load(open("config/config.json")); S = getSampleStyleSheet()
NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
PALE_BLUE = colors.HexColor("#EAF2F8")
INK = colors.HexColor("#243447")
MUTED = colors.HexColor("#5B6770")
H1 = ParagraphStyle("h1", S["Heading1"], fontName="Helvetica-Bold", fontSize=21, leading=25,
                    textColor=NAVY, spaceAfter=8)
H2 = ParagraphStyle("h2", S["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18,
                    textColor=NAVY, spaceBefore=12, spaceAfter=7, keepWithNext=True)
H3 = ParagraphStyle("h3", S["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                    textColor=BLUE, spaceBefore=8, spaceAfter=4, keepWithNext=True)
B = ParagraphStyle("b", S["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.4,
                   textColor=INK, spaceAfter=5)
META = ParagraphStyle("meta", B, fontSize=10.5, leading=14, textColor=MUTED)
CELL = ParagraphStyle("c", B, fontSize=7.8, leading=9.6, spaceAfter=0)
CELL_HEAD = ParagraphStyle("ch", CELL, fontName="Helvetica-Bold", textColor=colors.white)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9, backColor=PALE_BLUE,
                      borderColor=colors.HexColor("#D5E3F0"), borderWidth=.5, borderPadding=6)
def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t); t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return re.sub(r"(https?://[^\s<)]+)", r"<link href='\1' color='blue'>\1</link>", t)
def md_to_flow(md):
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            j = i + 1; code = []
            while not lines[j].startswith("```"): code.append(lines[j]); j += 1
            out.append(Preformatted("\n".join(code).replace("├── ","|-- ").replace("└── ","`-- ").replace("│","|"), CODE)); i = j + 1; continue
        if l.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-|:]+\|$", lines[i]): rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            n = max(len(r) for r in rows); rows = [r + [""] * (n - len(r)) for r in rows]
            W = 17 * cm; widths = [W / n] * n if n != 2 else [4.2 * cm, W - 4.2 * cm]
            table_rows = [
                [Paragraph(inline(c), CELL_HEAD if row_index == 0 else CELL) for c in r]
                for row_index, r in enumerate(rows)
            ]
            t = Table(table_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#B8C7D3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            out += [t, Spacer(1, 6)]; continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", l)
        if m and os.path.exists(m.group(2)):
            w, h = ImageReader(m.group(2)).getSize(); W = min(16 * cm, 16 * cm); out.append(Image(m.group(2), W, W * h / w)); i += 1; continue
        if l.startswith("# "): out.append(Paragraph(inline(l[2:]), H1))
        elif l.startswith("## "): out.append(Paragraph(inline(l[3:]), H2))
        elif l.startswith("### "): out.append(Paragraph(inline(l[4:]), H3))
        elif l.startswith(("* ", "- ")): out.append(Paragraph("• " + inline(l[2:]), B))
        elif l.strip(): out.append(Paragraph(inline(l), B))
        i += 1
    return out
doc = SimpleDocTemplate("submission/ML_Assignment2_Submission.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.7*cm, bottomMargin=1.7*cm,
                        title="ML Assignment 2 – Submission", author=cfg.get("student_name") or "Student")
gh = cfg.get("github_url") or "https://github.com/<your-username>/<repo-name>  (to be replaced)"
ap = cfg.get("app_url") or "https://<your-app>.streamlit.app  (to be replaced)"
identity = Table([[Paragraph("<b>Student</b>", CELL), Paragraph(inline(cfg.get("student_name") or "________"), B)],
                  [Paragraph("<b>BITS ID</b>", CELL), Paragraph(inline(cfg.get("bits_id") or "________"), B)]],
                 colWidths=[3.2 * cm, 13.8 * cm], hAlign="LEFT")
identity.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
    ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#B8C7D3")),
    ("INNERGRID", (0, 0), (-1, -1), .35, colors.HexColor("#D5E3F0")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
F = [Paragraph("Machine Learning", META), Paragraph("Assignment 2 | End-to-end classification and deployment", H1),
     identity, Spacer(1, 16),
     Paragraph("1. GitHub Repository Link", H2), Paragraph(inline(gh), B), Spacer(1, 5),
     Paragraph("2. Live Streamlit App Link", H2), Paragraph(inline(ap), B), Spacer(1, 5),
     Paragraph("3. BITS Virtual Lab usage screenshots", H2)]
shots = cfg.get("screenshot") or []
if isinstance(shots, str):
    shots = [shots] if shots else []
valid_shots = [shot for shot in shots if os.path.exists(os.path.join(ROOT, shot))]
if valid_shots:
    for shot in valid_shots:
        shot_path = os.path.join(ROOT, shot)
        w, h = ImageReader(shot_path).getSize()
        F.append(Image(shot_path, 16 * cm, 16 * cm * h / w))
        F.append(Spacer(1, 8))
else:
    t = Table([[Paragraph("<b>[ Add BITS Virtual Lab usage screenshots ]</b><br/>Set 'screenshot' in config.json and rebuild.", B)]], colWidths=[16*cm], rowHeights=[6*cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])); F.append(t)
F += [PageBreak(), Paragraph("4. README Content", H2), Spacer(1, 4)] + md_to_flow(open("README.md", encoding="utf-8").read())
def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5E3F0"))
    canvas.line(2 * cm, 1.25 * cm, 19 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, .85 * cm, "BITS Pilani WILP · M.Tech (AIML) · Machine Learning")
    canvas.drawRightString(19 * cm, .85 * cm, f"Page {document.page}")
    canvas.restoreState()
doc.build(F, onFirstPage=footer, onLaterPages=footer)
print("PDF built")
