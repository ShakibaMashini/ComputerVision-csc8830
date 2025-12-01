import cv2
import numpy as np
import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# These folders hold uploaded images, processed output, and the 10 template images.
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
TEMPLATE_FOLDER = "templates_db"

# Make sure the folders exist so the app won't crash the first time it runs.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------------------------------
# Load all template images from the template_db folder.
# Each template is stored as (filename, image).
# -------------------------------------------------------
def load_templates():
    templates = []
    for file in sorted(os.listdir(TEMPLATE_FOLDER)):
        path = os.path.join(TEMPLATE_FOLDER, file)
        img = cv2.imread(path)
        if img is not None:
            templates.append((file, img))  # save the name and image together
    return templates

templates = load_templates()


# -------------------------------------------------------
# Non-Maximum Suppression:
# This removes overlapping bounding boxes so we don't detect
# the same object multiple times just because the template
# matched in several nearby spots.
# -------------------------------------------------------
def non_max_suppression(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Extract coordinates of each bounding box
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort boxes by their score (highest confidence first)
    order = scores.argsort()[::-1]

    keep = []

    # Process boxes while there are still candidates left
    while order.size > 0:
        i = order[0]  # box with highest score
        keep.append(i)

        # Compute overlap between this box and the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep only the boxes that do not overlap too much
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


# -------------------------------------------------------
# Multi-scale template matching:
# We try each template at several different scales so that
# even if the object is larger or smaller in the scene,
# we can still find it.
#
# For every detected match, we blur that region and draw a box.
# -------------------------------------------------------
def process_template_matching(image_path):
    scene = cv2.imread(image_path)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    detected_boxes = []
    scores = []
    labels = []

    # This threshold decides how confident a match needs to be.
    # Higher = stricter, lower = more detections.
    MATCH_THRESHOLD = 0.84

    # Try each template
    for filename, temp in templates:
        temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        # Try the template at multiple scales (smaller to larger)
        for scale in np.linspace(0.5, 1.5, 15):
            resized = cv2.resize(temp_gray, None, fx=scale, fy=scale)
            tH, tW = resized.shape[:2]

            # Skip templates that are larger than the scene
            if tH > scene_gray.shape[0] or tW > scene_gray.shape[1]:
                continue

            # Perform template matching using normalized correlation
            result = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)

            # Find all locations where the match score is good enough
            yloc, xloc = np.where(result >= MATCH_THRESHOLD)

            for (x, y) in zip(xloc, yloc):
                detected_boxes.append([x, y, x + tW, y + tH])
                scores.append(result[y, x])
                labels.append(filename)

    # Remove overlapping boxes so each object is detected once
    keep = non_max_suppression(detected_boxes, scores)

    output = scene.copy()

    # Apply blur and draw boxes for each final detection
    for i in keep:
        x1, y1, x2, y2 = detected_boxes[i]
        label = labels[i]

        # Blur the inside of the box to hide the object
        roi = output[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (45, 45), 0)
        output[y1:y2, x1:x2] = blurred

        # Draw the red bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Write the filename of the template above the box
        cv2.putText(output, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save the processed image to the outputs folder
    filename = f"output_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, output)

    return output_path


# -------------------------------------------------------
# Flask routes (web API)
# -------------------------------------------------------

# Homepage
@app.route("/")
def home():
    return render_template("index.html")


# When the user uploads an image, process it here
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Run the template matching + blurring pipeline
    result_path = process_template_matching(upload_path)

    return jsonify({"output_url": f"/output/{os.path.basename(result_path)}"})


# Serve processed output images
@app.route("/output/<filename>")
def send_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/png")


# Run the web server
if __name__ == "__main__":
    app.run(debug=True)
