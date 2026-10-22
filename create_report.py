from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
ROOT=Path(__file__).resolve().parent
metrics=json.loads((ROOT/'models'/'metrics.json').read_text())
imp=__import__('pandas').read_csv(ROOT/'models'/'permutation_importance.csv')
out=ROOT/'reports'/'Aiqon_Technical_Report.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CoverTitle', parent=styles['Title'], fontSize=21, leading=26, textColor=colors.HexColor('#16324F'), alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name='Sub', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor('#526273'), spaceAfter=16))
styles.add(ParagraphStyle(name='H', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#16324F'), spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=8.5, leading=11, spaceAfter=4))
styles.add(ParagraphStyle(name='Body', parent=styles['BodyText'], fontSize=9.2, leading=13, spaceAfter=6))
def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.grey)
    canvas.drawString(18*mm, 12*mm, 'Aiqon AI/SE Intern Take-Home Challenge | Synthetic-data prototype')
    canvas.drawRightString(192*mm, 12*mm, f'Page {doc.page}')
    canvas.restoreState()
story=[]
story += [Spacer(1,14*mm), Paragraph('Explainable Shipment ETA Prediction<br/>and Delay Propagation',styles['CoverTitle']), Paragraph('Aiqon AI/SE Intern — Round 2 Technical Challenge',styles['Sub'])]
story += [Paragraph('<b>Executive summary</b>',styles['H']), Paragraph('This project implements a reproducible machine-learning pipeline that estimates final shipment delay, classifies the risk of a delay exceeding 24 hours, and examines which known-at-checkpoint features the model relies on. A synthetic dataset of 600 journeys is generated with explicit upstream/downstream delay interactions. A Random Forest regressor and classifier are evaluated on a held-out test set and compared with a median-delay baseline.',styles['Body'])]
story += [Paragraph('1. Problem framing and prediction checkpoint',styles['H']), Paragraph('The prediction checkpoint is defined as <b>after destination-port arrival but before customs clearance</b>. The model uses only features intended to be known at this point: route, route distance, planned transit/port/customs durations, observed departure delay, observed port-arrival delay, weather severity, port congestion, and customs complexity. Realized port-processing delay, customs delay, final arrival timestamp, and target columns are excluded from model inputs to reduce target leakage.',styles['Body'])]
story += [Paragraph('2. Dataset and delay propagation',styles['H']), Paragraph('The generator simulates 600 shipment journeys across five illustrative routes. Each row includes scheduled and actual timestamps, stage-level delays, external-condition indicators, and final delay. The synthetic process models port-processing delay as a function of congestion and observed port-arrival delay; customs delay depends on customs complexity, congestion, and port-processing delay. Final delay combines upstream and downstream effects plus stochastic variation. The binary risk label is final delay greater than 24 hours. These equations are designed assumptions, not estimated logistics laws.',styles['Body'])]
story += [Paragraph('3. Modelling and evaluation',styles['H']), Paragraph('A training-set median predictor provides a baseline. The primary regression model is a Random Forest with 350 trees and minimum leaf size 2. A separate Random Forest classifier estimates the probability of delay exceeding 24 hours. A fixed random 80/20 split is used, with stratification for classification. Regression metrics are MAE, RMSE, and R²; classification metrics are precision, recall, F1, ROC-AUC, and confusion matrix.',styles['Body'])]
reg=metrics['regression']; cls=metrics['classification']
data=[['Metric','Held-out result'],['Test records',str(metrics['test_rows'])],['Median baseline MAE',f"{reg['baseline_median_mae_hours']:.3f} hours"],['Random Forest MAE',f"{reg['mae_hours']:.3f} hours"],['Random Forest RMSE',f"{reg['rmse_hours']:.3f} hours"],['Regression R²',f"{reg['r2']:.3f}"],['Classifier precision / recall / F1',f"{cls['precision']:.3f} / {cls['recall']:.3f} / {cls['f1']:.3f}"],['Classifier ROC-AUC',str(cls['roc_auc'])]]
t=Table(data,colWidths=[82*mm,78*mm],repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#16324F')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CFD8E3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F9')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t, Spacer(1,4*mm), Paragraph('On this run, regression MAE improved relative to the training-median baseline. The classifier performance is measured on a held-out test set; because the data is synthetic, these metrics indicate internal consistency with the simulator rather than expected performance on real shipments.',styles['Small'])]
story += [PageBreak(), Paragraph('4. Explainability and findings',styles['H']), Paragraph('Held-out permutation importance measures how much test MAE worsens when a feature is shuffled. It is a global model-reliance measure, not proof that a feature causes delay. The top features in this run are shown below.',styles['Body'])]
rows=[['Feature','MAE increase after shuffling (hours)']]
for _,r in imp.head(6).iterrows(): rows.append([str(r['feature']),f"{r['importance_mae_increase_hours']:.3f} ± {r['std']:.3f}"])
t2=Table(rows,colWidths=[80*mm,80*mm],repeatRows=1)
t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#16324F')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CFD8E3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F9')]),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t2,Spacer(1,4*mm)]
for fname,caption in [('feature_importance.png','Figure 1. Held-out permutation importance.'),('actual_vs_predicted.png','Figure 2. Actual versus predicted final delay on held-out records.')]:
    fp=ROOT/'reports'/fname
    if fp.exists():
        story += [Image(str(fp),width=78*mm,height=49*mm),Paragraph(caption,styles['Small'])]
story += [Paragraph('5. Implementation and reproducibility',styles['H']), Paragraph('The repository contains separate modules for data generation, feature definitions/leakage validation, model training/evaluation, explanation plots, automated tests, and an optional Streamlit demo. Run commands from the repository root: `pip install -r requirements.txt`, `python src/generate_data.py`, `python src/train_model.py`, `python src/explainability.py`, `pytest -q`, and `streamlit run app.py`.',styles['Body'])]
story += [Paragraph('6. Limitations and future work',styles['H']), Paragraph('The dataset is synthetic, small, and generated from assumptions authored for this challenge; it cannot establish real-world generalization. A random split does not test temporal drift. Feature importance is not causal inference. The demo does not consume live AIS, weather, port, or customs data and does not produce calibrated uncertainty intervals. Next steps are to validate against real historical timestamps, compare chronological and route-held-out splits, calibrate probabilities, add prediction intervals, monitor data drift, and obtain logistics-domain review.',styles['Body'])]
story += [Paragraph('Conclusion',styles['H']), Paragraph('The prototype connects stage-aware synthetic data, a defined prediction checkpoint, leakage-aware feature selection, regression and classification, held-out evaluation, and interpretable model diagnostics. Its main contribution is a transparent end-to-end structure that can be validated and extended when real shipment data becomes available.',styles['Body'])]
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=15*mm,bottomMargin=20*mm,title='Explainable Shipment ETA Prediction and Delay Propagation')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
