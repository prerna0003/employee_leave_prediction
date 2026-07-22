import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the logistic regression model
MODEL_PATH = "logistic_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Feature mapping order expected by the model
FEATURE_NAMES = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Dashboard</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .dashboard-card {
            border: none;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            background: #ffffff;
        }
        .header-box {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border-radius: 16px 16px 0 0;
            padding: 2rem;
        }
        .form-label {
            font-weight: 600;
            color: #374151;
            font-size: 0.9rem;
        }
        .btn-predict {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border: none;
            padding: 12px;
            font-weight: 600;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(124, 58, 237, 0.4);
        }
        .result-badge {
            border-radius: 12px;
            padding: 1.5rem;
        }
    </style>
</head>
<body class="py-5">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card dashboard-card">
                    <!-- Dashboard Header -->
                    <div class="header-box text-center">
                        <h2 class="fw-bold mb-1"><i class="fa-solid me-2"></i>Employee Analytics Dashboard</h2>
                        <p class="mb-0 opacity-75">Logistic Regression Classification Model</p>
                    </div>

                    <div class="card-body p-4 p-md-5">
                        {% if error %}
                            <div class="alert alert-danger d-flex align-items-center" role="alert">
                                <i class="fa-solid fa-triangle-exclamation me-2"></i>
                                <div>{{ error }}</div>
                            </div>
                        {% endif %}

                        {% if prediction is not none %}
                            <div class="result-badge text-center mb-4 {% if prediction == 1 %}bg-warning-subtle text-warning-emphasis{% else %}bg-success-subtle text-success-emphasis{% endif %}">
                                <h5 class="text-uppercase fw-semibold mb-1">Prediction Outcome</h5>
                                <h2 class="display-6 fw-bold mb-0">Class Output: {{ prediction }}</h2>
                                {% if probability is not none %}
                                    <p class="mt-2 mb-0 fw-semibold">Model Confidence: {{ "%.2f"|format(probability * 100) }}%</p>
                                {% endif %}
                            </div>
                        {% endif %}

                        <form method="POST" action="/">
                            <div class="row g-4">
                                <!-- Education -->
                                <div class="col-md-6">
                                    <label class="form-label">Education Level</label>
                                    <select name="Education" class="form-select" required>
                                        <option value="0">Bachelors (0)</option>
                                        <option value="1">Masters (1)</option>
                                        <option value="2">PHD (2)</option>
                                    </select>
                                </div>

                                <!-- Joining Year -->
                                <div class="col-md-6">
                                    <label class="form-label">Joining Year</label>
                                    <input type="number" min="2000" max="2030" name="JoiningYear" class="form-control" value="2018" required>
                                </div>

                                <!-- City -->
                                <div class="col-md-6">
                                    <label class="form-label">City Category</label>
                                    <select name="City" class="form-select" required>
                                        <option value="0">Bangalore (0)</option>
                                        <option value="1">Pune (1)</option>
                                        <option value="2">New Delhi (2)</option>
                                    </select>
                                </div>

                                <!-- Payment Tier -->
                                <div class="col-md-6">
                                    <label class="form-label">Payment Tier (1-3)</label>
                                    <input type="number" min="1" max="3" name="PaymentTier" class="form-control" value="3" required>
                                </div>

                                <!-- Age -->
                                <div class="col-md-6">
                                    <label class="form-label">Age</label>
                                    <input type="number" min="18" max="70" name="Age" class="form-control" value="28" required>
                                </div>

                                <!-- Gender -->
                                <div class="col-md-6">
                                    <label class="form-label">Gender</label>
                                    <select name="Gender" class="form-select" required>
                                        <option value="0">Male (0)</option>
                                        <option value="1">Female (1)</option>
                                    </select>
                                </div>

                                <!-- Ever Benched -->
                                <div class="col-md-6">
                                    <label class="form-label">Ever Benched</label>
                                    <select name="EverBenched" class="form-select" required>
                                        <option value="0">No (0)</option>
                                        <option value="1">Yes (1)</option>
                                    </select>
                                </div>

                                <!-- Experience In Current Domain -->
                                <div class="col-md-6">
                                    <label class="form-label">Domain Experience (Years)</label>
                                    <input type="number" min="0" max="40" name="ExperienceInCurrentDomain" class="form-control" value="3" required>
                                </div>
                            </div>

                            <button type="submit" class="btn btn-primary btn-predict w-100 mt-4 text-white">
                                <i class="fa-solid fa-chart-line me-2"></i>Run Prediction
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction=None, probability=None, error="Model file not found on server.")

    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            features = [float(request.form[feat]) for feat in FEATURE_NAMES]
            input_data = np.array([features])
            
            # Perform Prediction
            pred_class = model.predict(input_data)[0]
            prediction = int(pred_class)

            # Get Confidence Score if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_data)[0]
                probability = float(probs[prediction])

        except Exception as e:
            error = f"Prediction Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, error=error)


@app.route("/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model file not found"}), 500

    try:
        data = request.get_json(force=True)
        features = [float(data[feat]) for feat in FEATURE_NAMES]
        input_data = np.array([features])
        
        prediction = int(model.predict(input_data)[0])
        response = {"prediction": prediction}
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            response["probability"] = float(probs[prediction])

        return jsonify(response)
    except KeyError as e:
        return jsonify({"error": f"Missing feature parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Required for Vercel Serverless integration
app = app# employee_leave_prediction
