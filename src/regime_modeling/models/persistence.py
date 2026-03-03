import os
import pickle
from datetime import datetime
from typing import Dict, Any

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from regime_modeling.config import (
    N_REGIMES,
    DEFAULT_HMM_PARAMS,
    MODELS_DIR
)


def save_best_model(
    model: GaussianHMM, 
    scaler: StandardScaler, 
    params: Dict[str, Any], 
    metrics: Dict[str, Any], 
    filename_prefix: str = 'best_hmm_model'
) -> Dict[str, str]:
    """Save HMM model, scaler, and configuration to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_params = {
        'n_components': N_REGIMES,
        'covariance_type': model.covariance_type,
        'n_iter': model.n_iter,
        'random_state': params.get('random_state', DEFAULT_HMM_PARAMS['random_state'])
    }

    complete_params = {**params, **model_params}
    
    model_path = f'{MODELS_DIR}/{filename_prefix}_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'params': complete_params,
            'metrics': metrics,
            'timestamp': timestamp
        }, f)

    latest_path = f'{MODELS_DIR}/{filename_prefix}_latest.pkl'
    with open(latest_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'params': complete_params,
            'metrics': metrics,
            'timestamp': timestamp
        }, f)

    return {
        'model_path': model_path,
        'latest_path': latest_path
    }
