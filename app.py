from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask import Flask, render_template, request, redirect, url_for, session
from flask import send_from_directory
from handwriting_features import extract_features, extract_devanagari_features
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------
# ✅ APP CONFIG
# --------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ✅ DATABASE MODELS
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200))


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    image_path = db.Column(db.String(200))
    processed_path = db.Column(db.String(200))
    neatness = db.Column(db.Float)
    spacing = db.Column(db.Float)
    consistency = db.Column(db.Float)
    overall = db.Column(db.Float)
    date = db.Column(db.String(50))
    weak_areas = db.Column(db.String(200))
    language = db.Column(db.String(20))

# ✅ Helper function
#English
def find_weaknesses(scores):
    weaknesses = []

    if scores.get("neatness", 100) < 60:
        weaknesses.append("neatness.html")

    if scores.get("spacing", 100) < 60:
        weaknesses.append("spacing.html")

    if scores.get("consistency", 100) < 60:
        weaknesses.append("consistency.html")

    return weaknesses

#Devanagari
def find_devanagari_weaknesses(scores):
    weaknesses = []

    if scores.get("shirorekha", 100) < 60:
        weaknesses.append("shirorekha")

    if scores.get("matra", 100) < 60:
        weaknesses.append("matra")

    if scores.get("samanta", 100) < 60:
        weaknesses.append("samanta")

    return weaknesses

# --------------------
# ✅ ROUTES
# --------------------
@app.route('/')
def home():
    return render_template('index.html')


# ✅ Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

	# Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "Email already exists! Try logging in instead."

        user = User(name=name, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')


# ✅ Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            return redirect(url_for('home'))
        else:
            return "Invalid credentials!"

    return render_template('login.html')


# ✅ Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# ✅ Upload Page
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    lang = request.form.get("language")
    if request.method == 'POST':
        file = request.files.get('file')
        language = request.form.get('language')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            return redirect(url_for('result', filename=filename, lang=language))

        return "No file selected!"
    return render_template('upload.html')


# ✅ Result Page
@app.route('/result/<filename>/<lang>')
def result(filename, lang):

    lang = lang.lower().strip()

    if lang in ["hindi", "marathi", "dev"]:
        lang = "devanagari"
    print("Debug: Language received:", lang)

    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(original_path)

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_" + filename)
    cv2.imwrite(processed_path, thresh)

    # ----------------------------
    # ✅ ENGLISH HANDWRITING MODEL
    # ----------------------------
    if lang in ["english","eng"]:
        features = extract_features(img)

        neatness = max(0, 100 - abs(features["slant_angle"]))
        spacing_score = max(0, 100 - abs(30 - features["avg_spacing"]))
        consistency_score = max(0, 100 - abs(40 - features["avg_letter_height"]))

        overall_score = (neatness + spacing_score + consistency_score) / 3

        scores = {
            "neatness": round(neatness, 1),
            "spacing": round(spacing_score, 1),
            "consistency": round(consistency_score, 1),
            "overall": round(overall_score, 1)
        }

        feedback = []
        if scores["neatness"] < 60:
            feedback.append("Your handwriting slants too much. Try keeping letters upright.")
        if scores["spacing"] < 60:
            feedback.append("Spacing between words is inconsistent.")
        if scores["consistency"] < 60:
            feedback.append("Letter height varies. Practice maintaining uniform letter size.")
        if len(feedback) == 0:
            feedback.append("Your English handwriting is excellent!")

        #Weak Areas
        weak_areas = []
        if scores["neatness"] < 60:
            weak_areas.append("neatness")
        if scores["spacing"] < 60:
            weak_areas.append("spacing")
        if scores["consistency"] < 60:
            weak_areas.append("consistency")

    # ----------------------------
    # ✅ DEVANAGARI HANDWRITING MODEL
    # ----------------------------
    elif lang in ["devanagari", "hindi", "marathi", "dev"]:
        features = extract_devanagari_features(img)

        shirorekha_score = min(100, max(0, features["shirorekha_strength"] * 100))
        matra_score = min(100, max(0, features["matra_score"] * 100))
        samanta_score = max(0, 100 - features["height_variation"])

        overall_score = (shirorekha_score + matra_score + samanta_score) / 3

        scores = {
            "shirorekha": round(shirorekha_score, 1),
            "matra": round(matra_score, 1),
            "samanta": round(samanta_score, 1),
            "overall": round(overall_score, 1)
        }

        feedback = []
        if scores["shirorekha"] < 60:
            feedback.append("Shirorekha (top line) is weak or broken. Try writing smoother top lines.")
        if scores["matra"] < 60:
            feedback.append("Matras are unclear or inconsistent.")
        if scores["samanta"] < 60:
            feedback.append("Letter height varies too much. Practice writing uniform characters.")
        if len(feedback) == 0:
            feedback.append("Your Devanagari handwriting is excellent!")

        #Weak Ares
        weak_areas = []
        if scores["shirorekha"] < 60:
            weak_areas.append("shirorekha")
        if scores["matra"] < 60:
            weak_areas.append("matra")
        if scores["samanta"] < 60:
            weak_areas.append("samanta")

    # ✅ Save report to database if logged in
    existing = Report.query.filter_by(image_path=f"static/uploads/{filename}").first()

    if lang == "english":
        neat = scores["neatness"]
        spac = scores["spacing"]
    else:
        neat = scores["shirorekha"]
        spac = scores["matra"]

    if 'user_id' in session:
       existing = Report.query.filter_by(
        user_id=session['user_id'],
        image_path=f"static/uploads/{filename}"
    ).first()

    if existing:
        report_id = existing.id
        new_report = existing
    else:
        new_report = Report(
            user_id=session['user_id'],
            image_path=f"static/uploads/{filename}",
            processed_path=f"static/uploads/processed_{filename}",
            neatness=scores.get('neatness', 0),
            spacing=scores.get('spacing', 0),
            consistency=scores['consistency'],
            overall=scores['overall'],
            weak_areas=",".join(weak_areas),
            language=lang,
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        db.session.add(new_report)
        db.session.commit()
        report_id = new_report.id
    
    # ✅ FINAL RETURN 
    return render_template(
        'result.html',
        image_path=f"static/uploads/{filename}",
        processed_image=f"static/uploads/processed_{filename}",
        features=features,
        scores=scores,
        feedback=feedback,
        lang=lang,
        report_id=new_report.id,
        weak_areas=weak_areas
    )


# ✅ View Report
@app.route('/view_report/<int:report_id>')
def view_report(report_id):
    report = Report.query.get(report_id)
    if not report:
        return "Report not found!"

    # Recreate scores dict
    if report.language == "english":
        scores = {
            "neatness": report.neatness,
            "spacing": report.spacing,
            "consistency": report.consistency,
            "overall": report.overall
        }
    else:
        scores = {
            "shirorekha": report.neatness,
            "matra": report.spacing,
            "consistency": report.consistency,
            "overall": report.overall
        }

    # Features are not saved in DB, so send empty dict
    features = {}

    # Decide language (simple detection)
    lang = report.language

    feedback = []
    if scores["neatness"] < 60:
        feedback.append("Your handwriting slants too much.")
    if scores["spacing"] < 60:
        feedback.append("Spacing between words is inconsistent.")
    if scores["consistency"] < 60:
        feedback.append("Letter height varies too much.")
    if len(feedback) == 0:
        feedback.append("Your handwriting is good!")


    return render_template(
        "result.html",
        image_path=report.image_path,
        processed_image=report.processed_path,
        features=features,
        scores=scores,
        feedback=feedback,
        lang=lang,
        report_id=report.id
    )



# ✅ Reports Page
@app.route('/reports')
def reports():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    user_reports = Report.query.filter_by(user_id=session['user_id']).all()

    for r in user_reports:

        # For English reports (values are real)
        if r.language =="english":
            scores = {
                "neatness": r.neatness,
                "spacing": r.spacing,
                "consistency": r.consistency
            }

        # For Marathi / Devanagari reports
        else:
            scores = {
                "shirorekha": r.neatness,
                "matra": r.spacing,
                "samanta": r.consistency
            }

        min_val = min(scores.values())
        r.weak_areas = [k for k, v in scores.items() if v == min_val]

    return render_template('reports.html', reports=user_reports)

# ✅ Download report PDF

@app.route('/download_report/<int:report_id>')
def download_report(report_id):
    report = Report.query.get(report_id)

    if not report:
        return "Report not found!"

    filename = f"report_{report_id}.pdf"
    filepath = os.path.join("static", "pdf_reports")

    # Create folder if it doesn't exist
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    full_path = os.path.join(filepath, filename)

    # Create PDF
    c = canvas.Canvas(full_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "AI Handwriting Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Date: {report.date}")
    c.drawString(50, 700, f"User ID: {report.user_id}")

    # Images
    try:
        c.drawImage(
            ImageReader(os.path.join(BASE_DIR, report.image_path)),
            50, 500, width=200, height=150
        )
        c.drawString(50, 660, "Original Handwriting")
    except:
        pass

    try:
        c.drawImage(ImageReader(report.processed_path), 300, 500, width=200, height=150)
        c.drawString(300, 660, "Processed Image")
    except:
        pass

    # Scores
    c.drawString(50, 450, f"Neatness: {report.neatness}%")
    c.drawString(50, 430, f"Spacing: {report.spacing}%")
    c.drawString(50, 410, f"Consistency: {report.consistency}%")
    c.drawString(50, 390, f"Overall Score: {report.overall}%")

    # Close PDF
    c.save()

    return redirect(f"/static/pdf_reports/{filename}")

# ✅ Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Fetch all reports for this user
    user_reports =Report.query.filter_by(user_id=user_id).order_by(Report.date.asc()).all()

    # If not user_reports yet
    if not user_reports:
        return render_template('dashboard.html', reports=[], scores=[])

    # Extract all scores
    overall_scores = [r.overall for r in user_reports]
    neatness_scores = [r.neatness for r in user_reports]
    spacing_scores = [r.spacing for r in user_reports]
    consistency_scores = [r.consistency for r in user_reports]

    # Total reports
    total_reports = len(user_reports)

     # Averages
    avg_neatness = sum(neatness_scores) / total_reports
    avg_spacing = sum(spacing_scores) / total_reports
    avg_consistency = sum(consistency_scores) / total_reports
    avg_overall = sum(overall_scores) / total_reports

    # Determine badge
    if total_reports >= 21:
        badge = "Expert"
        badge_color = "#d9534f"   # red
    elif total_reports >= 11:
        badge = "Advanced"
        badge_color = "#f0ad4e"   # orange
    elif total_reports >= 4:
        badge = "Intermediate"
        badge_color = "#5cb85c"   # green
    else:
        badge = "Beginner"
        badge_color = "#0275d8"   # blue

    # Badge Progress Logic
 
    if total_reports < 4:
        next_badge = "Intermediate"
        remaining = 4 - total_reports

    elif total_reports < 11:
        next_badge = "Advanced"
        remaining = 11 - total_reports

    elif total_reports < 21:
        next_badge = "Expert"
        remaining = 21 - total_reports

    else:
        next_badge = "Max Level Reached"
        remaining = 0



    # Best & Worst
    best_overall = max(overall_scores)
    worst_overall = min(overall_scores)

    # Improvement %
    if overall_scores[0] == 0:
        improvement = 0
    else:
        improvement = ((overall_scores[-1] - overall_scores[0]) / overall_scores[0]) * 100

    # AI Suggestions
    suggestions = []

    if improvement > 10:
        suggestions.append("Great job! Your handwriting has improved significantly.")
    elif improvement < -5:
        suggestions.append("Your score dropped recently. Try slowing down your writing and focusing on shapes.")
    else:
        suggestions.append("Your handwriting is stable. Try practicing curves and slants to improve further.")

    if avg_spacing < 40:
        suggestions.append("Your spacing is tight. Try practicing equal spacing between letters.")
    elif avg_spacing > 70:
        suggestions.append("Your spacing is wide. Try keeping letters closer to improve consistency.")

    if avg_consistency < 50:
        suggestions.append("Your letter consistency needs attention. Try rewriting the same word multiple times.")

    if avg_neatness < 50:
        suggestions.append("Work on your neatness. Practice writing slowly and cleanly for 5 minutes daily.")

    # Statistics
    stats = {
        "total_reports": total_reports,
        "best_score": best_overall,
        "average_score": avg_overall,
        "worst_score": worst_overall
    }

    # Dates for charts
    dates = [r.date for r in user_reports]


    # Prepare data for charts
    dates = [r.date for r in user_reports]
    overall_scores = [r.overall for r in user_reports]
    neatness_scores = [r.neatness for r in user_reports]
    spacing_scores = [r.spacing for r in user_reports]
    consistency_scores = [r.consistency for r in user_reports]

    return render_template(
        "dashboard.html",
        reports=user_reports,
        stats=stats,
        total_reports=total_reports,
        avg_neatness=avg_neatness,
        avg_spacing=avg_spacing,
        avg_consistency=avg_consistency,
        avg_overall=round(avg_overall,2),
        best_overall=best_overall,
        worst_overall=worst_overall,
        improvement=round(improvement,2),
        suggestions=suggestions,
        dates=dates,
        overall_scores=overall_scores,
        neatness_scores=neatness_scores,
        spacing_scores=spacing_scores,
        consistency_scores=consistency_scores,
        badge_color=badge_color,
        next_badge=next_badge,
        remaining=remaining,
    	badge=badge
    )

# ✅ Worksheets

@app.route("/worksheet", methods=["GET", "POST"])
def worksheet():
    if request.method == "GET":
        return render_template("worksheet_form.html")  
    heading = request.form.get("heading")
    is_double_line = request.form.get("double_line") == "on"

    user_text = request.form.get("text")
    text_lines = user_text.split("\n")

    lines = []
    for t in text_lines:
        lines.append({
            "sample": t.strip(),
            "type": "double" if is_double_line else "single"
        })

    return render_template("worksheet.html", heading=heading, lines=lines)


# ✅ Open Worksheets

@app.route("/worksheet/<filename>")
def open_worksheet(filename):
    return render_template(f"{filename }.html")

@app.route("/worksheet/devanagari/<file>")
def devanagari_worksheet(file):
    return render_template(f"dev_{file}.html")


# ✅ Full Page Practice 
@app.route("/full_practice/<text>")
def full_practice(text):
    return render_template("full_page_practice.html",
                           heading="Full Page Practice",
                           trace_text=text)

@app.route("/practice")
def practice_menu():
    return render_template("practice_menu.html")


@app.route("/worksheet/english/az")
def english_az_sheet():
    lines = [ "A a", "B b", "C c", "D d", "E e", 
              "F f", "G g", "H h", "I i", "J j",
              "K k", "L l", "M m", "N n", "O o",
              "P p", "Q q", "R r", "S s", "T t",
              "U u", "V v", "W w", "X x", "Y y", "Z z" ]

    return render_template("full_page_practice.html",
                           heading="English A–Z Practice Sheet",
                           lines=lines[:12])  # fits full page


@app.route("/worksheet/cursive")
def cursive_sheet():
    cursive_set = [
        "aa bb cc", 
        "dd ee ff",
        "gg hh ii",
        "jj kk ll",
        "mm nn oo",
        "pp qq rr",
        "ss tt uu",
        "vv ww xx",
        "yy zz"
    ]

    return render_template("full_page_practice.html",
                           heading="Cursive Practice Sheet",
                           lines=cursive_set[:12])


@app.route("/worksheet/devanagari/Matra")
def devanagari_matra_sheet():
    lines = ["का", "कि", "की", "कु", "कू", "के", "कै", "को", "कौ", "कं", "क:"] * 2

    return render_template("full_page_practice.html",
                           heading="Devanagari Matra Practice",
                           lines=lines[:12])


@app.route("/worksheet/devanagari/Shirorekha")
def devanagari_shirorekha_sheet():
    lines = ["क", "ख", "ग", "घ", "त", "न", "म", "फ", "थ", "ध", "भ", "श"]

    return render_template("full_page_practice.html",
                           heading="Shirorekha Practice Sheet",
                           lines=lines[:12])



# ✅ Delete Report

@app.route('/delete_report/<int:report_id>', methods=['POST'])
def delete_report(report_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    report = Report.query.filter_by(id=report_id, user_id=session['user_id']).first()

    if not report:
        return "Report not found or unauthorized!", 404


    # Delete image files
    try:
        if os.path.exists(report.image_path):
            os.remove(report.image_path)
        if os.path.exists(report.processed_path):
            os.remove(report.processed_path)
    except:
        pass

    # Delete DB record
    db.session.delete(report)
    db.session.commit()

    return redirect(url_for('reports'))



# ✅ Run App
if __name__ == '__main__':
    app.run(debug=True)
