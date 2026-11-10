"""Create downloadable PDF reports for individual DiaPredict predictions."""

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)


# Display labels are kept separate from database column names so the PDF
# remains readable without changing the existing database schema.
FEATURE_LABELS = [
    ("pregnancies", "Pregnancies"),
    ("glucose", "Glucose"),
    ("blood_pressure", "Blood Pressure"),
    ("skin_thickness", "Skin Thickness"),
    ("insulin", "Insulin"),
    ("bmi", "BMI"),
    ("diabetes_pedigree_function", "Diabetes Pedigree Function"),
    ("age", "Age"),
]


def _safe_number(value, decimals=2):
    """Format a numeric value safely for display in the PDF."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError, OverflowError):
        return "Not available"


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=colors.HexColor("#123047"),
        alignment=TA_CENTER,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitleCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#557080"),
        alignment=TA_CENTER,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeadingCustom",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#087E8B"),
        spaceBefore=10,
        spaceAfter=7,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="SmallNoteCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
    ))
    return styles


def _draw_page_footer(canvas, document):
    """Draw a consistent footer and page number on every PDF page."""
    canvas.saveState()
    page_width, _ = A4
    y = 11 * mm

    canvas.setStrokeColor(colors.HexColor("#D5E0E5"))
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, y + 5 * mm,
                page_width - document.rightMargin, y + 5 * mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#607580"))
    canvas.drawString(document.leftMargin, y, "DiaPredict | Academic Research Report")
    canvas.drawRightString(
        page_width - document.rightMargin,
        y,
        f"Page {document.page}",
    )
    canvas.restoreState()


def generate_prediction_report_pdf(prediction, explanation=None) -> bytes:
    """Return a PDF report as bytes; no temporary files are created."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=22 * mm,
        title=f"DiaPredict Prediction Report {prediction.id}",
        author="DiaPredict",
    )
    styles = _build_styles()
    story = [
        Paragraph("DiaPredict", styles["ReportTitleCustom"]),
        Paragraph(
            "Individual Prediction Report | Academic Research System",
            styles["ReportSubtitleCustom"],
        ),
    ]

    created_at = prediction.created_at
    if isinstance(created_at, datetime):
        created_text = created_at.strftime("%d %B %Y, %I:%M %p")
    else:
        created_text = str(created_at or "Not available")

    predicted_class = int(prediction.predicted_class)
    predicted_probability = float(prediction.predicted_probability)
    probability_class_1 = (
        predicted_probability if predicted_class == 1
        else 1.0 - predicted_probability
    )
    probability_class_0 = 1.0 - probability_class_1
    classification = (
        "Higher model-estimated diabetes risk (Class 1)"
        if predicted_class == 1
        else "Lower model-estimated diabetes risk (Class 0)"
    )

    summary_rows = [
        ["Prediction ID", str(prediction.id)],
        ["Generated on", created_text],
        ["Model output class", str(predicted_class)],
        ["Model classification", classification],
        ["Probability — Class 1", f"{probability_class_1 * 100:.2f}%"],
        ["Probability — Class 0", f"{probability_class_0 * 100:.2f}%"],
    ]
    story.append(Paragraph("Prediction Summary", styles["SectionHeadingCustom"]))
    summary_table = Table(
        summary_rows,
        colWidths=[55 * mm, 112 * mm],
        hAlign="LEFT",
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3F6")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#203744")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([summary_table, Spacer(1, 7)])

    story.append(Paragraph(
        "Submitted Input Parameters",
        styles["SectionHeadingCustom"],
    ))
    feature_rows = [["Feature", "Recorded value"]]
    for attribute, label in FEATURE_LABELS:
        value = getattr(prediction, attribute, None)
        decimals = 3 if attribute == "diabetes_pedigree_function" else 2
        feature_rows.append([label, _safe_number(value, decimals)])

    feature_table = Table(
        feature_rows,
        colWidths=[83 * mm, 84 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    feature_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F7F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DE")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([feature_table, Spacer(1, 7)])

    if explanation is not None:
        shap_rows = [["Feature", "SHAP contribution"]]
        contributions = explanation.feature_contributions or {}

        if isinstance(contributions, dict) and contributions:
            def contribution_sort_value(item):
                try:
                    return abs(float(item[1]))
                except (TypeError, ValueError, OverflowError):
                    return -1

            sorted_items = sorted(
                contributions.items(),
                key=contribution_sort_value,
                reverse=True,
            )
            for feature, contribution in sorted_items:
                shap_rows.append([
                    str(feature).replace("_", " ").title(),
                    _safe_number(contribution, 5),
                ])
        else:
            shap_rows.append([
                "No feature contribution data",
                "Not available",
            ])

        shap_table = Table(
            shap_rows,
            colWidths=[83 * mm, 84 * mm],
            repeatRows=1,
            hAlign="LEFT",
        )
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087E8B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F3F7F9")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DE")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        # Keep the heading, baseline/output values, complete SHAP table, and
        # explanatory note together. If they do not fit in the remaining
        # space, ReportLab moves the whole block to the next page.
        shap_block = [
            Paragraph("SHAP Explanation", styles["SectionHeadingCustom"]),
            Paragraph(
                f"Baseline value: {_safe_number(explanation.base_value, 6)}",
                styles["SmallNoteCustom"],
            ),
            Paragraph(
                f"Model output: {_safe_number(explanation.model_output, 6)}",
                styles["SmallNoteCustom"],
            ),
            Spacer(1, 4),
            shap_table,
            Spacer(1, 4),
            Paragraph(
                "SHAP contributions describe how the model's output is "
                "attributed across the input features. They do not establish "
                "causation.",
                styles["SmallNoteCustom"],
            ),
        ]
        story.append(KeepTogether(shap_block))
    else:
        story.extend([
            Paragraph("SHAP Explanation", styles["SectionHeadingCustom"]),
            Paragraph(
                "A stored SHAP explanation is not available for this "
                "prediction record.",
                styles["SmallNoteCustom"],
            ),
        ])

    disclaimer = (
        "MEDICAL DISCLAIMER: DiaPredict is an academic machine-learning "
        "research project. Its output is an estimate produced by a statistical "
        "model, not a medical diagnosis. This report must not be used as a "
        "substitute for professional medical advice, diagnosis, or treatment. "
        "Consult a qualified healthcare professional for medical decisions."
    )
    story.append(Spacer(1, 12))
    story.append(KeepTogether([
        Paragraph("Important Disclaimer", styles["SectionHeadingCustom"]),
        Paragraph(disclaimer, styles["SmallNoteCustom"]),
    ]))

    document.build(
        story,
        onFirstPage=_draw_page_footer,
        onLaterPages=_draw_page_footer,
    )
    return buffer.getvalue()


def generate_research_report_pdf() -> bytes:
    """Generate the complete DiaPredict independent-test research report."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=22 * mm,
        title="DiaPredict Research Report",
        author="DiaPredict",
    )
    styles = _build_styles()

    metrics = [
        ["Metric", "Independent test result", "Interpretation"],
        ["Accuracy", "72.08%", "Overall correct predictions"],
        ["Precision", "58.46%", "Positive prediction precision"],
        ["Recall", "70.37%", "Positive cases identified"],
        ["Specificity", "73.00%", "Negative cases identified"],
        ["F1 Score", "63.87%", "Harmonic mean of precision and recall"],
        ["ROC-AUC", "80.85%", "Area under the ROC curve"],
        ["Classification Threshold", "0.35", "Selected operating threshold"],
    ]


    confusion = [
        ["Actual / Predicted", "Predicted Negative", "Predicted Positive"],
        ["Actual Negative", "73\nTrue Negative", "27\nFalse Positive"],
        ["Actual Positive", "16\nFalse Negative", "38\nTrue Positive"],
    ]

    dataset_rows = [
        ["Dataset information", "Value"],
        ["Dataset", "Pima Indians Diabetes"],
        ["Total records", "768"],
        ["Input features", "8"],
        ["Training records", "614"],
        ["Independent test records", "154"],
    ]
    methodology_rows = [
        ["Evaluation methodology", "Value"],
        ["Selected model", "Logistic Regression"],
        ["Data splitting", "Stratified Split"],
        ["Random state", "42"],
        ["Training strategy", "Training-Only Tuning"],
        ["Threshold selection", "Training OOF"],
    ]

    findings = [
        (
            "Recall Performance",
            "The model achieved 70.37% recall, identifying 38 of the 54 actual positive cases in the independent test set."
        ),
        (
            "False Positive Observations",
            "The model produced 27 false positive predictions. This indicates that some negative cases were incorrectly classified as positive."
        ),
        (
            "Model Discrimination",
            "The ROC-AUC score of 80.85% indicates the model's ability to distinguish between the two classes across different classification thresholds."
        ),
        (
            "Explainable AI Integration",
            "SHAP analysis is integrated to explain how individual input features contribute to model outputs."
        ),
    ]
    limitations = [
        "The evaluation uses a relatively small independent test set of 154 records.",
        "The model has not undergone clinical validation or prospective testing.",
        "The threshold selection procedure may introduce selection optimism; nested cross-validation could provide a more rigorous evaluation.",
        "The dataset results may not generalize to every population or healthcare setting.",
    ]

    story = [
        Paragraph("DiaPredict", styles["ReportTitleCustom"]),
        Paragraph(
            "Research Evaluation Report | Independent Test Performance",
            styles["ReportSubtitleCustom"],
        ),
        Paragraph("1. Research Overview", styles["SectionHeadingCustom"]),
        Paragraph(
            "This report summarizes the independent test evaluation of the "
            "DiaPredict Logistic Regression model using 154 unseen test records. "
            "The reported metrics and methodology correspond to the Research "
            "Reports page in the application.",
            styles["Normal"],
        ),
        Paragraph("Model Evaluation Metrics", styles["SectionHeadingCustom"]),
    ]

    metric_table = Table(metrics, colWidths=[34 * mm, 39 * mm, 94 * mm], repeatRows=1)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DE")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([metric_table, Spacer(1, 5)])

    story.append(Paragraph("2. Confusion Matrix", styles["SectionHeadingCustom"]))
    story.append(Paragraph(
        "The matrix compares actual class labels with model-predicted class labels.",
        styles["Normal"],
    ))
    confusion_table = Table(confusion, colWidths=[55 * mm, 56 * mm, 56 * mm], repeatRows=1)
    confusion_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#E2F5EF")),
        ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#FFF1D8")),
        ("BACKGROUND", (1, 2), (1, 2), colors.HexColor("#FBE8ED")),
        ("BACKGROUND", (2, 2), (2, 2), colors.HexColor("#E1F5FA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CAD8DE")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([confusion_table, Spacer(1, 5)])

    story.append(Paragraph("3. Dataset and Methodology", styles["SectionHeadingCustom"]))
    story.append(Paragraph("Dataset Information", styles["Heading3"]))
    dataset_table = Table(dataset_rows, colWidths=[83 * mm, 84 * mm], repeatRows=1)
    dataset_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DE")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([dataset_table, Spacer(1, 7)])
    methodology_table = Table(methodology_rows, colWidths=[83 * mm, 84 * mm], repeatRows=1)
    methodology_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087E8B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DE")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    # Keep the subsection heading and its table on the same page.
    story.append(KeepTogether([
        Paragraph("Evaluation Methodology", styles["Heading3"]),
        methodology_table,
    ]))

    story.append(Paragraph("4. Research Findings", styles["SectionHeadingCustom"]))
    for heading, description in findings:
        story.append(Paragraph(f"<b>{heading}</b>", styles["Normal"]))
        story.append(Paragraph(description, styles["Normal"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("5. Research Limitations", styles["SectionHeadingCustom"]))
    for limitation in limitations:
        story.append(Paragraph(f"&bull; {limitation}", styles["Normal"]))
        story.append(Spacer(1, 5))

    disclaimer = (
        "<b>Medical disclaimer:</b> DiaPredict is an academic machine-learning "
        "research project. Its predictions are not medical diagnoses and must "
        "not replace professional medical advice. The model has not undergone "
        "clinical validation or prospective testing."
    )
    story.extend([
        Spacer(1, 13),
        Paragraph("Important Disclaimer", styles["SectionHeadingCustom"]),
        Paragraph(disclaimer, styles["SmallNoteCustom"]),
    ])

    document.build(
        story,
        onFirstPage=_draw_page_footer,
        onLaterPages=_draw_page_footer,
    )
    return buffer.getvalue()
