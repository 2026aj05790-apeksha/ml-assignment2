"""Build the single submission PDF: (1) GitHub link (2) Streamlit link (3) BITS-Lab screenshot (4) README content."""
import json, os, re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether
from reportlab.lib.utils import ImageReader
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
cfg = json.load(open("config.json")); S = getSampleStyleSheet()
H1 = ParagraphStyle("h1", S["Heading1"], textColor=colors.HexColor("#1f3a5f")); H2 = ParagraphStyle("h2", S["Heading2"], textColor=colors.HexColor("#1f3a5f"))
B = ParagraphStyle("b", S["BodyText"], fontSize=9, leading=12); CELL = ParagraphStyle("c", B, fontSize=7.5, leading=9.5)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.2)
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
            t = Table([[Paragraph(inline(c), CELL) for c in r] for r in rows], colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe6f4")), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            out += [t, Spacer(1, 6)]; continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", l)
        if m and os.path.exists(m.group(2)):
            w, h = ImageReader(m.group(2)).getSize(); W = min(16 * cm, 16 * cm); out.append(Image(m.group(2), W, W * h / w)); i += 1; continue
        if l.startswith("# "): out.append(Paragraph(inline(l[2:]), H1))
        elif l.startswith("## "): out.append(Paragraph(inline(l[3:]), H2))
        elif l.startswith("### "): out.append(Paragraph("<b>" + inline(l[4:]) + "</b>", S["Heading3"]))
        elif l.startswith(("* ", "- ")): out.append(Paragraph("• " + inline(l[2:]), B))
        elif l.strip(): out.append(Paragraph(inline(l), B))
        i += 1
    return out
doc = SimpleDocTemplate("submission/ML_Assignment2_Submission.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="ML Assignment 2 – Submission", author=cfg.get("student_name") or "Student")
gh = cfg.get("github_url") or "https://github.com/<your-username>/<repo-name>  (to be replaced)"
ap = cfg.get("app_url") or "https://<your-app>.streamlit.app  (to be replaced)"
F = [Paragraph("BITS Pilani – M.Tech (AIML) – Machine Learning – Assignment 2", H1),
     Paragraph(f"Name: {cfg.get('student_name') or '________'} &nbsp;&nbsp; BITS ID: {cfg.get('bits_id') or '________'}", B), Spacer(1, 14),
     Paragraph("1. GitHub Repository Link", H2), Paragraph(inline(gh), B), Spacer(1, 10),
     Paragraph("2. Live Streamlit App Link", H2), Paragraph(inline(ap), B), Spacer(1, 10),
     Paragraph("3. Screenshot – assignment execution on BITS Virtual Lab", H2)]
shot = cfg.get("screenshot")
if shot and os.path.exists(shot):
    w, h = ImageReader(shot).getSize(); F.append(Image(shot, 16 * cm, 16 * cm * h / w))
else:
    t = Table([[Paragraph("<b>[ Insert ONE BITS Virtual Lab screenshot here ]</b><br/>Set 'screenshot' in config.json and rebuild.", B)]], colWidths=[16*cm], rowHeights=[6*cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])); F.append(t)
F += [PageBreak(), Paragraph("4. README Content", H2), Spacer(1, 4)] + md_to_flow(open("README.md", encoding="utf-8").read())
doc.build(F, onFirstPage=lambda c, d: None, onLaterPages=lambda c, d: (c.setFont("Helvetica", 7), c.drawRightString(19 * cm, 1 * cm, f"Page {d.page}")))
print("PDF built")
