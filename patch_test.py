import sys
with open("api.py", "r") as f:
    content = f.read()
content = content.replace("stats = index.describe_index_stats()", "stats = index.describe_index_stats().to_dict()")
with open("api.py", "w") as f:
    f.write(content)
