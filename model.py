import numpy as np 
import plotly.graph_objects as go 
import pandas as pd
from scipy.stats import norm
from features import get_merged_data
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def run_hmm_model():

    data = get_merged_data(join_type='inner')   
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(data)


    states = ["Bull Market", "Bear Market", "Sideways Market"]
    n_states = len(states)

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500, random_state = 42)
    model.fit(data)

    log_probability, hidden_states = model.decode(obs_scaled,
                                              lengths = len(obs_scaled),
                                              algorithm ='viterbi' )

    print('Log Probability :',log_probability)
    print("Most likely hidden states:", hidden_states)


def main(): 
    run_hmm_model()

if __name__ == "__main__":
    main()