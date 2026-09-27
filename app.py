import os
import sys

# Add the root directory to the system path to allow importing from 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the actual Streamlit app located in src/app.py
import src.app
