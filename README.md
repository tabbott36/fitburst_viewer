# fitburst — Initial Guess GUI

A standalone, browser-based viewer for setting initial parameter guesses
before running fitburst on your fast radio burst data.

## Usage

Just open the HTML file in any modern browser — no server or dependencies required:

```bash
open fitburst_initial_guess.html          # macOS
xdg-open fitburst_initial_guess.html     # Linux
start fitburst_initial_guess.html        # Windows
```

## Workflow

1. **Step 1 — Arrival time**: Move your mouse over the dynamic spectrum.
   A green dashed line tracks your cursor. Click to lock in the arrival time.

2. **Step 2 — Width**: Click and drag horizontally across the burst to define
   the pulse width (σ). The yellow shaded region shows the selected interval.

3. **Step 3 — Copy output**: The dictionary in the output panel updates live.
   Click "Copy to clipboard" to grab the JSON for use in your fitburst call.

## Plugging in real data

In the `<script>` section of the HTML, replace the `genDynspec()` function
with your own data. The array should have shape `[n_freq][n_time]` with
values normalised to [0, 1]:

```javascript
// Replace genDynspec() with something like:
function genDynspec() {
  // dynspec is your [n_freq x n_time] array, values in [0, 1]
  const rows = dynspec.length;
  const cols  = dynspec[0].length;
  return { data: dynspec, rows, cols };
}
```

If you're loading data from a file, you can use a small Python helper to
export your numpy array as JSON and fetch it in the browser:

```python
import numpy as np, json
dynspec = ...  # your (n_freq, n_time) array
normed  = (dynspec - dynspec.min()) / dynspec.ptp()
with open('dynspec.json', 'w') as f:
    json.dump(normed.tolist(), f)
```

Then in the HTML, replace the `genDynspec()` call with a `fetch('dynspec.json')`
and re-render on load.

## Output format

```json
{"arrival_time": 84.5000, "width": 12.3000}
```

Units are milliseconds, matching the fitburst convention for `t0` and `sigma`.
