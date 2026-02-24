import cv2
import numpy as np

def extract_features(img):

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Slant Detection
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    slant_angle = 0
    if lines is not None:
        for line in lines[:10]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            slant_angle += angle
        slant_angle /= len(lines[:10])

    # 2. Stroke Thickness
    kernel = np.ones((3, 3), np.uint8)
    thick = cv2.dilate(thresh, kernel, iterations=1)
    stroke_thickness = np.mean(thick / 255)

    # 3. Letter Height
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 10]
    avg_letter_height = np.mean(heights) if heights else 0

    # 4. Spacing (between contours)
    xs = sorted([cv2.boundingRect(c)[0] for c in contours])
    gaps = np.diff(xs)
    avg_spacing = np.mean(gaps) if len(gaps) > 1 else 0

    return {
        "slant_angle": float(slant_angle),
        "stroke_thickness": float(stroke_thickness),
        "avg_letter_height": float(avg_letter_height),
        "avg_spacing": float(avg_spacing)
    }

def extract_devanagari_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Feature 1: Shirorekha Presence (Top Line)
    row_sum = np.sum(thresh[:20, :])
    shirorekha_strength = row_sum / (thresh.shape[1] * 255)

    # Feature 2: Matra Detection (vertical marks)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))
    matras = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    matra_score = np.sum(matras) / (thresh.size * 255)

    # Feature 3: Character Height Consistency
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 20]
    height_variation = np.std(heights) if len(heights) > 2 else 0

    return {
        "shirorekha_strength": float(shirorekha_strength),
        "matra_score": float(matra_score),
        "height_variation": float(height_variation)
    }

