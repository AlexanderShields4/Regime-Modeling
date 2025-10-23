import yfinance
import pandas as pd
import streamlit as st
import numpy as np
import hmmlearn
from hmmlearn.hmm import MultinomialHMM

print("yfinance:", yfinance.__version__)
print("pandas:", pd.__version__)
print("streamlit:", st.__version__)
print("numpy:", np.__version__)
print("hmmlearn:", hmmlearn.__version__)
