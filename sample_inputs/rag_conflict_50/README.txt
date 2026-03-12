Topic to use: RAG

Expected behavior:
- At least one cluster should include both supporting and contradicting claims.
- If exactly one support source and one contradict source land in the same cluster,
  confidence for that cluster should be 50%.

Tips:
- Start with clustering distance threshold around 0.40.
- If support/contradict claims split into separate clusters, raise threshold to 0.50-0.60.
