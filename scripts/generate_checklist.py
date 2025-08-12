# writes to docs/data/latest.json so GitHub Pages can serve it
import json, os
# ... build your chain here (same as notebook) ...
result = chain.invoke({"role":"hr","level":"junior","hire_type":"external","task":"Onboard ..."})
os.makedirs("docs/data", exist_ok=True)
json.dump(result, open("docs/data/latest.json","w"), ensure_ascii=False, indent=2)
