  import re

  with open('/home/ubuntu/thesis/data/processed/covid.json', 'r') as f:
      covid = json.load(f)

  for country in covid:
      covid[country].pop("vaccination rate", None)  # cleaner deletion

  # First, dump to a pretty string
  json_str = json.dumps(covid, indent=4)

  # Then compact any lists (bracketed blocks) onto a single line
  json_str = re.sub(r'\[\s+([^]]*?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', json_str)

  # Save to file
  with open('covid.json', 'w') as f:
      f.write(json_str)