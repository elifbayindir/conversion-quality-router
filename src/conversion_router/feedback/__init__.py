"""Local human-feedback and audit loop.

Records what the model said, what the decision agent (or its fallback)
recommended, and what a human reviewer decided, as append-only rows in a local
SQLite file. It is evidence for later, human-led policy review only: nothing
here retrains the model, changes the threshold, or alters the decision policy.
"""
