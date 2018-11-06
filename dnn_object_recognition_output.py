import json
import sys

classification_results = sys.argv[1]

results = []

for line in classification_results.split(','):
	line = line.split(':')
	if '' not in line:
		results.append({
			'classification': line[1].strip(),
			'confidence': line[0]
		});

print json.dumps(results)
