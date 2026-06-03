import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tempfile
import os
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Bank Statement Analyser",
    page_icon="🏦",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 1.8rem 0;
        background: linear-gradient(135deg, #0a0f2c 0%, #0d1b4b 50%, #0a2463 100%);
        border-radius: 14px;
        margin-bottom: 2rem;
        border: 1px solid #1e3a8a;
    }
    .main-header h1 { color: #f59e0b; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { color: #93c5fd; margin: 0.4rem 0 0; font-size: 1rem; }
    .main-header .flag { font-size: 1.4rem; }

    .metric-card {
        background: #0f172a;
        border: 1px solid #1e3a8a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label {
        color: #64748b;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .metric-card .value {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: .2rem;
    }

    .eligible-box {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #34d399;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .not-eligible-box {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #f87171;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-icon { font-size: 3rem; }
    .result-text { font-size: 1.7rem; font-weight: 700; margin: .4rem 0; }
    .result-sub  { font-size: 0.95rem; color: #cbd5e1; }

    .bank-badge {
        display: inline-block;
        background: #1e3a8a;
        border: 1px solid #3b82f6;
        color: #93c5fd;
        border-radius: 20px;
        padding: .3rem 1rem;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .insight-card {
        background: #0f172a;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    .stDataFrame { border-radius: 8px; }
    footer { visibility: hidden; }
    .stTabs [data-baseweb="tab"] { color: #93c5fd; }
    .stTabs [aria-selected="true"] { color: #f59e0b !important; border-bottom: 2px solid #f59e0b !important; }
</style>
""", unsafe_allow_html=True)

# ─── Indian Bank Detection Config ────────────────────────────────────────────
BANK_PATTERNS = {
    'State Bank of India':        ['state bank of india', 'sbi', 'sbin'],
    'Punjab National Bank':       ['punjab national bank', 'pnb', 'punb'],
    'Bank of India':              ['bank of india', 'boi', 'bkid'],
    'Bank of Baroda':             ['bank of baroda', 'bob', 'barb'],
    'Canara Bank':                ['canara bank', 'canara', 'cnrb'],
    'Union Bank of India':        ['union bank of india', 'union bank', 'ubin'],
    'Indian Bank':                ['indian bank', 'idib'],
    'Central Bank of India':      ['central bank of india', 'central bank', 'cbin'],
    'Indian Overseas Bank':       ['indian overseas', 'iob', 'ioba'],
    'UCO Bank':                   ['uco bank', 'ucba'],
    'Bank of Maharashtra':        ['bank of maharashtra', 'mahb'],
    'Punjab & Sind Bank':         ['punjab and sind', 'psib'],
    'HDFC Bank':                  ['hdfc bank', 'hdfc', 'hdfc0'],
    'ICICI Bank':                 ['icici bank', 'icici', 'icic'],
    'Axis Bank':                  ['axis bank', 'axis', 'utib'],
    'Kotak Mahindra Bank':        ['kotak mahindra', 'kotak', 'kkbk'],
    'IndusInd Bank':              ['indusind', 'indb'],
    'Yes Bank':                   ['yes bank', 'yesb'],
    'IDBI Bank':                  ['idbi bank', 'ibkl'],
    'Federal Bank':               ['federal bank', 'fdrl'],
    'South Indian Bank':          ['south indian bank', 'sibl'],
    'Karnataka Bank':             ['karnataka bank', 'karb'],
    'Bandhan Bank':               ['bandhan bank', 'bdbl'],
    'RBL Bank':                   ['rbl bank', 'ratn'],
    'AU Small Finance Bank':      ['au small finance', 'aubl'],
    'Jana Small Finance Bank':    ['jana small finance', 'jsfb'],
    'Paytm Payments Bank':        ['paytm payments bank', 'pytm'],
    'Airtel Payments Bank':       ['airtel payments bank', 'airp'],
    'NSDL Payments Bank':         ['nsdl payments', 'nspb'],
}

# ─── Training Data (Indian bank salary ranges in INR) ────────────────────────
TRAINING_DATA = {
    'total_credits': [
        340000, 420000, 185000, 295000, 510000, 130000, 480000, 310000,
        220000, 375000, 260000, 445000, 155000, 390000, 330000, 415000,
        240000, 340000, 385000, 200000, 320000, 475000, 280000, 360000,
        195000, 330000, 420000, 255000, 345000, 440000, 170000, 330000,
        390000, 215000, 325000, 430000, 285000, 335000, 390000, 185000,
        340000, 425000, 255000, 335000, 345000, 420000, 185000, 335000,
        395000, 215000, 325000,
    ],
    'total_debits': [
        240000, 330000, 225000, 170000, 375000, 178000, 305000, 238000,
        282000, 274000, 198000, 325000, 190000, 282000, 252000, 325000,
        289000, 241000, 292000, 219000, 198000, 335000, 231000, 271000,
        238000, 268000, 318000, 195000, 264000, 348000, 208000, 238000,
        292000, 219000, 241000, 327000, 243000, 264000, 289000, 211000,
        277000, 318000, 195000, 252000, 266000, 313000, 207000, 252000,
        295000, 219000, 241000,
    ],
    'num_transactions': [
        34, 26, 21, 29, 42, 19, 38, 33, 22, 35, 31, 40, 15, 37, 28,
        45, 23, 33, 30, 18, 35, 41, 29, 39, 21, 30, 42, 28, 34, 44,
        18, 31, 36, 24, 29, 40, 33, 31, 37, 19, 33, 39, 29, 32, 36,
        43, 20, 31, 37, 24, 29,
    ],
    'avg_transaction_amount': [
        9500, 15200, 11100, 12600, 11200, 9800, 10500, 10300, 12200, 10500,
        10000, 11600, 12600, 10700, 10900, 8600, 11400, 9000, 11800, 12600,
        10100, 11100, 9000, 10000, 11300, 10800, 10700, 9500, 10300, 9300,
        11300, 9400, 10300, 10600, 10100, 9400, 8400, 10300, 10100, 11400,
        10100, 11400, 10200, 9500, 9900, 10400, 10400, 9800, 11100, 10500, 10100,
    ],
    'transaction_variability': [
        14850, 12430, 8900, 13480, 17280, 9350, 16860, 11910, 9840, 13780,
        11540, 15890, 9820, 14520, 12430, 17500, 10780, 13810, 14420, 8550,
        13010, 15810, 12650, 14060, 10330, 12360, 16610, 11350, 13240, 17190,
        8910, 13190, 14660, 10030, 13060, 16270, 12280, 12940, 14560, 9260,
        13800, 16540, 11550, 13240, 13810, 16440, 8910, 13060, 14590, 9960, 13060,
    ],
    'balance_trend': [
        50000, 80000, -20000, 70000, 55000, -18000, 45000, 67000, -30000, 52000,
        40000, 53000, -32000, 43000, 46000, 55000, -29000, 48000, 51000, -22000,
        47000, 59000, 43000, 53000, -34000, 45000, 55000, 40000, 50000, 56000,
        -27000, 47000, 51000, -26000, 50000, 56000, 45000, 48000, 51000, -32000,
        50000, 55000, 42000, 45000, 50000, 56000, -27000, 45000, 53000, -26000, 50000,
    ],
    'Eligibility (y)': [
        1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,
        0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1,
    ],
}

# ─── ML Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.DataFrame(TRAINING_DATA)
    X = df[['total_credits','total_debits','num_transactions',
            'avg_transaction_amount','transaction_variability','balance_trend']]
    y = df['Eligibility (y)']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc


def identify_bank(text: str) -> str:
    t = text.lower()
    for bank, patterns in BANK_PATTERNS.items():
        if any(p in t for p in patterns):
            return bank
    return "Unknown Indian Bank"


def extract_text(pdf_bytes) -> str:
    text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# ─── Indian Transaction Categorization ───────────────────────────────────────
def categorize(description: str) -> str:
    d = description.lower()
    # Salary / Income
    if any(k in d for k in ['salary', 'sal cr', 'payroll', 'wages', 'remuneration', 'stipend']):
        return 'Salary / Income'
    # UPI / Digital Payments
    if any(k in d for k in ['upi', 'gpay', 'phonepe', 'paytm', 'bhim', 'razorpay', 'upi cr', 'upi dr']):
        return 'UPI & Digital Payments'
    # NEFT / RTGS / IMPS
    if any(k in d for k in ['neft', 'rtgs', 'imps', 'neft cr', 'neft dr', 'rtgs cr', 'rtgs dr']):
        return 'Transfers (NEFT/RTGS/IMPS)'
    # EMI / Loan
    if any(k in d for k in ['emi', 'loan', 'home loan', 'car loan', 'personal loan', 'emi dr', 'equated']):
        return 'EMI & Loan Payments'
    # ATM
    if any(k in d for k in ['atm', 'cash withdrawal', 'cash deposit', 'atm cash']):
        return 'ATM / Cash'
    # POS / Shopping
    if any(k in d for k in ['pos', 'amazon', 'flipkart', 'myntra', 'meesho', 'reliance', 'dmart', 'big bazaar',
                              'shopping', 'purchase', 'swiggy', 'zomato', 'blinkit', 'zepto', 'bigbasket', 'instamart']):
        return 'Shopping & E-commerce'
    # Utilities
    if any(k in d for k in ['electricity', 'bescom', 'msedcl', 'water bill', 'gas', 'indane', 'hp gas',
                              'mahanagar gas', 'igl', 'broadband', 'jiofiber', 'tata play', 'dth', 'dish tv']):
        return 'Utilities & Bills'
    # Telecom
    if any(k in d for k in ['jio', 'airtel', 'vi', 'vodafone', 'bsnl', 'mtnl', 'recharge', 'mobile bill',
                              'postpaid', 'prepaid', 'telecom']):
        return 'Mobile & Telecom'
    # Bank charges
    if any(k in d for k in ['service charge', 'service fee', 'bank charge', 'gst', 'processing fee',
                              'maintenance charge', 'sms charge', 'annual fee', 'demat']):
        return 'Bank Charges & GST'
    # Insurance
    if any(k in d for k in ['lic', 'life insurance', 'health insurance', 'star health', 'bajaj allianz',
                              'hdfc life', 'sbi life', 'max life', 'premium', 'insurance']):
        return 'Insurance & LIC'
    # Investment / Mutual Fund
    if any(k in d for k in ['sip', 'mutual fund', 'mf', 'zerodha', 'groww', 'kuvera', 'demat', 'nps',
                              'ppf', 'nsc', 'investment', 'fd', 'rd', 'fixed deposit', 'recurring']):
        return 'Investments & Savings'
    # Education
    if any(k in d for k in ['school', 'college', 'university', 'tuition', 'education', 'fees', 'hostel']):
        return 'Education'
    # Healthcare
    if any(k in d for k in ['hospital', 'clinic', 'pharmacy', 'apollo', 'fortis', 'medplus', 'healthcare',
                              'medical', 'doctor', 'diagnostic', 'lab test']):
        return 'Healthcare'
    # Fuel / Transport
    if any(k in d for k in ['petrol', 'diesel', 'fuel', 'iocl', 'bpcl', 'hpcl', 'pump', 'metro', 'irctc',
                              'railway', 'bus', 'cab', 'ola', 'uber', 'toll', 'fastag']):
        return 'Transport & Fuel'
    # Subscriptions
    if any(k in d for k in ['netflix', 'hotstar', 'amazon prime', 'sony liv', 'zee5', 'spotify', 'youtube',
                              'subscription', 'disney']):
        return 'OTT & Subscriptions'
    # Interest / Dividend
    if any(k in d for k in ['interest', 'int cr', 'dividend', 'int paid']):
        return 'Interest & Dividend'
    # Failed
    if any(k in d for k in ['failed', 'declined', 'unsuccessful', 'returned', 'insufficient', 'unpaid', 'dishonoured', 'bounce']):
        return 'Failed / Returned'
    # Credits
    if any(k in d for k in ['credit', 'cr ', 'deposit', 'received', 'inward']):
        return 'Credits & Deposits'
    return 'Other'


# ─── Transaction Parser ───────────────────────────────────────────────────────
def parse_transactions(text: str, bank: str) -> pd.DataFrame:
    rows = []

    def clean(s: str) -> float:
        return float(
            s.replace(',', '').replace('INR', '').replace('Rs.', '')
             .replace('Rs', '').replace('₹', '').replace(' ', '')
        )

    # ── Pattern A (PRIMARY): our generated PDFs + most Indian internet-banking exports
    # Format: DD Mon YYYY  DESCRIPTION  INR 1,23,456.78  Cr/Dr  INR 1,23,456.78
    patA = re.compile(
        r'^(\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})'
        r'\s+(.+?)'
        r'\s+INR\s+([\d,]+\.\d{2})'
        r'\s+(Cr|Dr)'
        r'\s+INR\s+([\d,]+\.\d{2})\s*$',
        re.IGNORECASE
    )

    # ── Pattern B: DD Mon YYYY  desc  amount(no INR prefix)  Cr/Dr  balance
    patB = re.compile(
        r'^(\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})'
        r'\s+(.+?)'
        r'\s+([\d,]+\.\d{2})'
        r'\s+(Cr|Dr)'
        r'\s+([\d,]+\.\d{2})\s*$',
        re.IGNORECASE
    )

    # ── Pattern C: DD/MM/YYYY or DD-MM-YYYY  desc  amount  Cr/Dr  balance
    patC = re.compile(
        r'^(\d{2}[/\-]\d{2}[/\-]\d{4})'
        r'\s+(.+?)'
        r'\s+([\d,]+\.\d{2})'
        r'\s+(Cr|Dr)'
        r'\s+([\d,]+\.\d{2})\s*$',
        re.IGNORECASE
    )

    # ── Pattern D: YYYY-MM-DD  desc  signed-amount  balance  (ISO, e.g. Axis/Kotak)
    patD = re.compile(
        r'^(\d{4}-\d{2}-\d{2})'
        r'\s+(.+?)'
        r'\s+(-?(?:INR\s*)?[\d,]+\.\d{2})'
        r'\s+(-?(?:INR\s*)?[\d,]+\.\d{2})\s*$',
        re.IGNORECASE
    )

    # ── Pattern E: DD/MM/YYYY  desc  signed-amount  balance  (no Cr/Dr tag)
    patE = re.compile(
        r'^(\d{2}[/\-]\d{2}[/\-]\d{4})'
        r'\s+(.+?)'
        r'\s+(-?[\d,]+\.\d{2})'
        r'\s+(-?[\d,]+\.\d{2})\s*$'
    )

    # ── Pattern F: loose search fallback — date anywhere in line
    patF = re.compile(
        r'(\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})'
        r'\s+(.+?)\s+INR\s+([\d,]+\.\d{2})\s+(Cr|Dr)\s+INR\s+([\d,]+\.\d{2})',
        re.IGNORECASE
    )

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if len(line) < 20:
            continue
        # skip header / footer lines
        if any(skip in line.lower() for skip in
               ['date', 'description', 'balance', 'statement period',
                'opening bal', 'closing bal', 'account summary',
                'transaction details', 'copyright', 'toll free']):
            continue

        matched = False

        for pat, n_groups, has_crdr in [
            (patA, 5, True),
            (patB, 5, True),
            (patC, 5, True),
        ]:
            m = pat.match(line)
            if m:
                date, desc, amt_s, crdr, bal_s = m.groups()
                try:
                    amt = clean(amt_s) if crdr.lower() == 'cr' else -clean(amt_s)
                    rows.append({'Date': date.strip(), 'Description': desc.strip(),
                                 'Amount': amt, 'Balance': clean(bal_s)})
                    matched = True
                    break
                except Exception:
                    pass

        if matched:
            continue

        for pat in (patD, patE):
            m = pat.match(line)
            if m:
                date, desc, amt_s, bal_s = m.groups()
                try:
                    rows.append({'Date': date.strip(), 'Description': desc.strip(),
                                 'Amount': clean(amt_s), 'Balance': clean(bal_s)})
                    matched = True
                    break
                except Exception:
                    pass

        if matched:
            continue

        # Last resort: loose search anywhere in line
        m = patF.search(line)
        if m:
            date, desc, amt_s, crdr, bal_s = m.groups()
            try:
                amt = clean(amt_s) if crdr.lower() == 'cr' else -clean(amt_s)
                rows.append({'Date': date.strip(), 'Description': desc.strip(),
                             'Amount': amt, 'Balance': clean(bal_s)})
            except Exception:
                pass

    df = pd.DataFrame(rows, columns=['Date', 'Description', 'Amount', 'Balance'])
    if not df.empty:
        df = df.drop_duplicates(subset=['Date', 'Description', 'Amount'])
        df['Category'] = df['Description'].apply(categorize)
        df['Type'] = df['Amount'].apply(lambda x: 'Credit' if x >= 0 else 'Debit')
    return df


def compute_features(df: pd.DataFrame) -> dict:
    credits = df[df['Amount'] > 0]['Amount']
    debits  = df[df['Amount'] < 0]['Amount']
    return {
        'total_credits':           credits.sum() if not credits.empty else 0,
        'total_debits':            abs(debits.sum()) if not debits.empty else 0,
        'num_transactions':        len(df),
        'avg_transaction_amount':  df['Amount'].abs().mean() if not df.empty else 0,
        'transaction_variability': df['Amount'].std() if len(df) > 1 else 0,
        'balance_trend':           (df['Balance'].iloc[-1] - df['Balance'].iloc[0]) if len(df) > 1 else 0,
    }


def predict_eligibility(model, features: dict):
    X = pd.DataFrame([features])[['total_credits','total_debits','num_transactions',
                                   'avg_transaction_amount','transaction_variability','balance_trend']]
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return int(pred), proba


def format_inr(val):
    """Format number as Indian Rupees"""
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    elif abs(val) >= 1e5:
        return f"₹{val/1e5:.2f} L"
    else:
        return f"₹{val:,.2f}"


def generate_insights(df, features, eligible, credit_ratio):
    insights = []
    # Salary regularity
    sal_rows = df[df['Category'] == 'Salary / Income']
    if len(sal_rows) > 0:
        avg_sal = sal_rows['Amount'].mean()
        insights.append(f"💼 Salary credits detected: avg {format_inr(avg_sal)} per credit.")
    # EMI burden
    emi_rows = df[df['Category'] == 'EMI & Loan Payments']
    if len(emi_rows) > 0:
        total_emi = abs(emi_rows['Amount'].sum())
        emi_to_income = (total_emi / max(features['total_credits'], 1)) * 100
        status = "✅ Healthy" if emi_to_income < 40 else "⚠️ High"
        insights.append(f"🏦 EMI burden: {format_inr(total_emi)} ({emi_to_income:.1f}% of credits) — {status}.")
    # Investment habit
    inv_rows = df[df['Category'] == 'Investments & Savings']
    if len(inv_rows) > 0:
        insights.append(f"📈 Investments/SIPs detected — positive financial discipline.")
    # Failed transactions
    fail_rows = df[df['Category'] == 'Failed / Returned']
    if len(fail_rows) > 0:
        insights.append(f"⚠️ {len(fail_rows)} failed/returned transaction(s) found — may affect eligibility.")
    # UPI usage
    upi_rows = df[df['Category'] == 'UPI & Digital Payments']
    if len(upi_rows) > 0:
        insights.append(f"📱 {len(upi_rows)} UPI/digital payment(s) — good digital footprint.")
    # Credit ratio
    insights.append(f"📊 Credit-to-debit ratio: {credit_ratio:.2f}x (threshold: 1.25x).")
    return insights


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="flag">🇮🇳</div>
  <h1>Indian Bank Statement Analyser</h1>
  <p>AI-powered loan eligibility assessment for all major Indian banks</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "Upload any Indian bank statement (PDF). "
        "Supports PSU, Private, Small Finance & Payments Banks. "
        "Detects your bank, parses transactions with Indian formats (UPI, NEFT, RTGS, IMPS), "
        "and runs an ML loan eligibility model."
    )
    st.header("🏦 Supported Banks")
    col1, col2 = st.columns(2)
    banks = list(BANK_PATTERNS.keys())
    half = len(banks) // 2
    with col1:
        for b in banks[:half]:
            st.markdown(f"<small>• {b}</small>", unsafe_allow_html=True)
    with col2:
        for b in banks[half:]:
            st.markdown(f"<small>• {b}</small>", unsafe_allow_html=True)
    st.divider()
    st.caption("Model: Random Forest (200 estimators, depth=6)")
    st.caption("Currency: Indian Rupees (INR ₹)")
    st.divider()
    st.header("📋 Eligibility Criteria")
    st.markdown("""
    - ✅ Credit-to-debit ratio ≥ 1.25×
    - ✅ Positive balance trend
    - ✅ ML model assessment
    - ✅ Regular income credits
    - ❌ Excessive failed transactions
    """)

# Load model
model, model_acc = train_model()

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload your Bank Statement (PDF)",
    type="pdf",
    help="Supports all Indian public, private, small finance, and payments banks"
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:3rem; background:#0f172a; border-radius:12px; border:2px dashed #1e3a8a; margin:1rem 0;">
      <div style="font-size:5rem;">📄</div>
      <h3 style="color:#64748b;">Drop your bank statement here</h3>
      <p style="color:#475569;">Supported: All major Indian banks · Format: PDF</p>
      <p style="color:#334155; font-size:0.85rem;">SBI · PNB · BOI · BOB · Canara · HDFC · ICICI · Axis · Kotak · and more</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Process ──────────────────────────────────────────────────────────────────
with st.spinner("🔍 Analysing your statement…"):
    # Read bytes ONCE — Streamlit UploadedFile.seek() is unreliable across versions.
    # Calling .read() a second time after .seek(0) can return empty bytes,
    # producing empty text → empty df → false "no transactions" warning.
    pdf_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    raw_text  = extract_text(io.BytesIO(pdf_bytes))
    bank_name = identify_bank(raw_text)
    df        = parse_transactions(raw_text, bank_name)

# ─── Header Row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"**Detected Bank:**<br><span class='bank-badge'>{bank_name}</span>", unsafe_allow_html=True)
with c2:
    st.markdown(f"**File:** `{uploaded_file.name}`")
with c3:
    st.markdown(f"**Model Accuracy:** `{model_acc*100:.1f}%`")
with c4:
    st.markdown(f"**Transactions Parsed:** `{len(df)}`")

st.divider()

# ─── No transactions fallback ────────────────────────────────────────────────
if df.empty:
    st.warning(
        "⚠️ No transactions could be parsed from this PDF. "
        "This may happen with image-based (scanned) PDFs or non-standard layouts. "
        "Try a text-selectable PDF exported from your bank's internet banking portal."
    )
    with fitz.open(tmp_path) as doc:
        full_text = "".join(page.get_text() for page in doc)

    # Try aggregate extraction (common in Indian bank statements)
    cb  = re.search(r'(?:Closing|Closing\s+Balance)\s*[:\-]?\s*([\d,]+\.?\d*)', full_text, re.IGNORECASE)
    tcr = re.search(r'(?:Total\s+Credits?|Credit\s+Total)\s*[:\-]?\s*([\d,]+\.?\d*)', full_text, re.IGNORECASE)
    ct  = re.search(r'(?:No\.?\s*of\s+Credit|Credit\s+Transactions?)\s*[:\-]?\s*(\d+)', full_text, re.IGNORECASE)

    if cb and tcr:
        closing = float(cb.group(1).replace(',',''))
        total_cr = float(tcr.group(1).replace(',',''))
        n_cr = int(ct.group(1)) if ct else 10

        feats = {
            'total_credits':           total_cr,
            'total_debits':            closing,
            'num_transactions':        n_cr,
            'avg_transaction_amount':  total_cr / max(n_cr, 1),
            'transaction_variability': total_cr * 0.15,
            'balance_trend':           closing,
        }
        pred, proba = predict_eligibility(model, feats)
        if pred:
            st.markdown('<div class="eligible-box"><div class="result-icon">✅</div>'
                        '<div class="result-text" style="color:#34d399;">Eligible for Loan</div>'
                        '<div class="result-sub">Based on aggregate statement data</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="not-eligible-box"><div class="result-icon">❌</div>'
                        '<div class="result-text" style="color:#f87171;">Not Eligible for Loan</div>'
                        '<div class="result-sub">Based on aggregate statement data</div></div>',
                        unsafe_allow_html=True)

    os.unlink(tmp_path)
    st.stop()

# ─── Full Analysis ────────────────────────────────────────────────────────────
features     = compute_features(df)
pred, proba  = predict_eligibility(model, features)
credit_ratio = features['total_credits'] / max(features['total_debits'], 1)
rule_pass    = credit_ratio >= 1.25 and features['balance_trend'] >= 0
eligible     = bool(pred) and rule_pass
confidence   = proba[1] * 100 if eligible else proba[0] * 100

# ── Eligibility Result ────────────────────────────────────────────────────────
st.subheader("📊 Loan Eligibility Result")
if eligible:
    st.markdown(
        f'<div class="eligible-box">'
        f'<div class="result-icon">✅</div>'
        f'<div class="result-text" style="color:#34d399;">Eligible for Loan</div>'
        f'<div class="result-sub">ML Confidence: {confidence:.1f}% &nbsp;·&nbsp; Credit Ratio: {credit_ratio:.2f}× &nbsp;·&nbsp; Balance Trend: Positive</div>'
        f'</div>', unsafe_allow_html=True)
else:
    reasons = []
    if credit_ratio < 1.25:
        reasons.append(f"Credit/Debit ratio {credit_ratio:.2f}× (min 1.25×)")
    if features['balance_trend'] < 0:
        reasons.append("Declining balance trend")
    if not bool(pred):
        reasons.append(f"ML model: {proba[1]*100:.1f}% eligible probability (min ~50%)")
    st.markdown(
        f'<div class="not-eligible-box">'
        f'<div class="result-icon">❌</div>'
        f'<div class="result-text" style="color:#f87171;">Not Eligible for Loan</div>'
        f'<div class="result-sub">{"&nbsp; · &nbsp;".join(reasons)}</div>'
        f'</div>', unsafe_allow_html=True)

st.divider()

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.subheader("💰 Key Financial Metrics")
m1,m2,m3,m4,m5,m6 = st.columns(6)

def mc(col, label, value):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True)

mc(m1, "Total Credits",   format_inr(features['total_credits']))
mc(m2, "Total Debits",    format_inr(features['total_debits']))
mc(m3, "Transactions",    str(features['num_transactions']))
mc(m4, "Avg Amount",      format_inr(features['avg_transaction_amount']))
mc(m5, "Std Dev",         format_inr(features['transaction_variability']))
mc(m6, "Balance Trend",   f"{'▲' if features['balance_trend'] >= 0 else '▼'} {format_inr(abs(features['balance_trend']))}")

st.divider()

# ── Insights ──────────────────────────────────────────────────────────────────
st.subheader("💡 Financial Insights")
insights = generate_insights(df, features, eligible, credit_ratio)
for ins in insights:
    st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("📈 Transaction Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Timeline", "🥧 Categories", "📊 Bar Chart", "⚖️ Credit vs Debit", "🔢 Data Table"])

DARK = 'plotly_dark'

with tab1:
    fig = px.line(df, x='Date', y='Amount', color='Category',
                  title='Transaction Timeline by Category',
                  template=DARK, markers=True)
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    if df['Balance'].notna().any():
        fig2 = px.area(df, x='Date', y='Balance', title='Running Balance',
                       template=DARK, color_discrete_sequence=['#34d399'])
        fig2.add_hline(y=0, line_dash='dash', line_color='#f87171', annotation_text='Zero')
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        cat = df.groupby('Category')['Amount'].sum().abs().reset_index()
        fig_p = px.pie(cat, values='Amount', names='Category',
                       title='Spend & Income by Category',
                       template=DARK, hole=0.38)
        fig_p.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_p, use_container_width=True)
    with col2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=credit_ratio,
            title={'text': "Credit-to-Debit Ratio", 'font': {'color': '#e2e8f0'}},
            delta={'reference': 1.25, 'increasing': {'color':'#34d399'}, 'decreasing': {'color':'#f87171'}},
            gauge={
                'axis': {'range':[0,3], 'tickcolor':'#64748b'},
                'bar':  {'color': '#34d399' if credit_ratio >= 1.25 else '#f87171'},
                'bgcolor': '#0f172a',
                'bordercolor': '#1e3a8a',
                'steps': [
                    {'range':[0, 1.25], 'color':'#1a0a0a'},
                    {'range':[1.25, 3], 'color':'#0a1a0f'},
                ],
                'threshold': {'line':{'color':'#f59e0b','width':3}, 'thickness':0.75, 'value':1.25}
            }
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=350)
        st.plotly_chart(fig_g, use_container_width=True)

    # Category breakdown table
    cat_table = df.groupby('Category').agg(
        Net_Amount=('Amount','sum'),
        Count=('Amount','count'),
        Avg=('Amount', lambda x: x.abs().mean())
    ).reset_index()
    cat_table['Net_Amount'] = cat_table['Net_Amount'].map(lambda v: f"₹{v:+,.2f}")
    cat_table['Avg']        = cat_table['Avg'].map(lambda v: f"₹{v:,.2f}")
    cat_table.columns       = ['Category','Net Amount','# Transactions','Avg Amount']
    st.dataframe(cat_table, use_container_width=True)

with tab3:
    fig_b = px.bar(df, x='Date', y='Amount', color='Category',
                   title='Amount per Transaction', template=DARK, barmode='relative')
    fig_b.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig_b, use_container_width=True)

    fig_cat = px.bar(
        df.groupby('Category')['Amount'].sum().reset_index(),
        x='Category', y='Amount', color='Category',
        title='Net Amount by Category', template=DARK
    )
    fig_cat.update_layout(showlegend=False, xaxis_tickangle=-35)
    st.plotly_chart(fig_cat, use_container_width=True)

with tab4:
    cr_amt = features['total_credits']
    dr_amt = features['total_debits']
    fig_cd = go.Figure()
    fig_cd.add_trace(go.Bar(name='Credits', x=['Credits'], y=[cr_amt],
                            marker_color='#34d399', text=[format_inr(cr_amt)], textposition='outside'))
    fig_cd.add_trace(go.Bar(name='Debits', x=['Debits'], y=[dr_amt],
                            marker_color='#f87171', text=[format_inr(dr_amt)], textposition='outside'))
    fig_cd.update_layout(template=DARK, title='Credits vs Debits', barmode='group', showlegend=True)
    st.plotly_chart(fig_cd, use_container_width=True)

    # Waterfall
    cat_net = df.groupby('Category')['Amount'].sum().sort_values()
    fig_wf = go.Figure(go.Waterfall(
        name="Net Flow", orientation="v",
        x=cat_net.index.tolist(),
        y=cat_net.values.tolist(),
        connector={"line": {"color": "#334155"}},
        increasing={'marker':{'color':'#34d399'}},
        decreasing={'marker':{'color':'#f87171'}},
        totals={'marker':{'color':'#f59e0b'}},
    ))
    fig_wf.update_layout(template=DARK, title='Waterfall: Net Flow by Category', showlegend=False)
    st.plotly_chart(fig_wf, use_container_width=True)

with tab5:
    st.markdown(f"**{len(df)} transactions parsed**")

    def color_amount(val):
        if isinstance(val, float):
            return 'color: #34d399' if val > 0 else 'color: #f87171'
        return ''

    st.dataframe(
        df.style.applymap(color_amount, subset=['Amount']),
        use_container_width=True,
        height=500
    )
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "transactions.csv", "text/csv")

# ── Feature Importance ───────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 Model Feature Importance")
feat_names = ['Total Credits','Total Debits','# Transactions','Avg Amount','Variability','Balance Trend']
imps = model.feature_importances_
fig_imp = px.bar(x=imps, y=feat_names, orientation='h',
                 title='Random Forest Feature Importances',
                 template=DARK, color=imps, color_continuous_scale='Blues',
                 labels={'x':'Importance','y':'Feature'})
fig_imp.update_layout(coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_imp, use_container_width=True)

# ── Probability Gauge ────────────────────────────────────────────────────────
st.divider()
st.subheader("🎯 ML Eligibility Probability")
col_a, col_b = st.columns(2)
with col_a:
    fig_prob = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba[1]*100,
        title={'text': "Eligibility Probability (%)", 'font': {'color': '#e2e8f0'}},
        gauge={
            'axis': {'range':[0,100], 'tickcolor':'#64748b'},
            'bar':  {'color': '#34d399' if proba[1] >= 0.5 else '#f87171'},
            'bgcolor': '#0f172a',
            'steps': [
                {'range':[0,50],  'color':'#1a0a0a'},
                {'range':[50,100],'color':'#0a1a0f'},
            ],
            'threshold': {'line':{'color':'#f59e0b','width':3},'thickness':0.75,'value':50}
        }
    ))
    fig_prob.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=300)
    st.plotly_chart(fig_prob, use_container_width=True)
with col_b:
    st.markdown("### 📋 Decision Summary")
    checks = [
        ("Credit/Debit Ratio ≥ 1.25×", credit_ratio >= 1.25, f"{credit_ratio:.2f}×"),
        ("Positive Balance Trend",       features['balance_trend'] >= 0, format_inr(features['balance_trend'])),
        ("ML Model Assessment",          bool(pred), f"{proba[1]*100:.1f}% probability"),
        ("Sufficient Transactions",      features['num_transactions'] >= 10, f"{features['num_transactions']} txns"),
        ("Credits Exceed Debits",        features['total_credits'] > features['total_debits'],
         f"{format_inr(features['total_credits'])} vs {format_inr(features['total_debits'])}"),
    ]
    for label, passed, detail in checks:
        icon  = "✅" if passed else "❌"
        color = "#34d399" if passed else "#f87171"
        st.markdown(f"<span style='color:{color}'>{icon} **{label}**</span> — {detail}", unsafe_allow_html=True)

os.unlink(tmp_path)