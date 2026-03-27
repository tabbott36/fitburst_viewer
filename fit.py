import sys

if len(sys.argv) != 3:
	print("Usage: python3 fit.py <arrival_time> <width>", file=sys.stderr)
	sys.exit(2)

try:
	arrival = float(sys.argv[1])
	width = float(sys.argv[2])
except ValueError:
	print("arrival_time and width must be numeric.", file=sys.stderr)
	sys.exit(2)

if width < 0:
	print("width must be non-negative.", file=sys.stderr)
	sys.exit(2)

print("Running fit with:")
print(f"arrival_time = {arrival}")
print(f"width = {width}")

# TODO: plug into fitburst here