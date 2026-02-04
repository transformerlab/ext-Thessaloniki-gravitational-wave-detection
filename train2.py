#!/usr/bin/env python3
import json
import tempfile
import os
from lab import lab

# Initialize lab
lab.init()

# Create and save a small artifact
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write("test artifact")
    lab.save_artifact(f.name, "test_artifact.txt")

# Create and save a fake dataset
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump({"data": [1, 2, 3]}, f)
    lab.save_artifact(f.name, "fake_dataset.json", type="dataset")

# Create and save a fake model
with tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False) as f:
    f.write("fake model data")
    lab.save_artifact(f.name, "fake_model.pt", type="model")
