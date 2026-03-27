import os
from flask import Flask, request, render_template, jsonify
import subprocess

app = Flask(__name__)

# ── serve GUI ──
@app.route('/')
def index():
    return render_template('index.html')

from flask import send_from_directory

@app.route('/observation.json')
def obs():
    return send_from_directory('.', 'observation.json')


# ── run fit ──
@app.route('/run-fit', methods=['POST'])
def run_fit():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        arrival_raw = data.get("arrival_time")
        width_raw = data.get("width")

        if arrival_raw is None or width_raw is None:
            return jsonify({"error": "Both 'arrival_time' and 'width' are required."}), 400

        try:
            arrival = float(arrival_raw)
            width = float(width_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "'arrival_time' and 'width' must be numeric."}), 400

        if width < 0:
            return jsonify({"error": "'width' must be non-negative."}), 400

        result = subprocess.run(
            ["python3", "fit.py", str(arrival), str(width)],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            return jsonify({
                "error": "fit.py failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }), 500

        output = result.stdout.strip()
        return jsonify({"output": output}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)