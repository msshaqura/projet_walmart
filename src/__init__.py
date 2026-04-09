# ------------------------------
# FICHIER : src/__init__.py
# (fichier vide mais nécessaire)
# ------------------------------

# Ce fichier indique que 'src' est un module Python
# Il peut rester vide ou contenir:

from .data_preprocessing import prepare_data
from .train_models import train_linear_regression, train_random_forest

__all__ = [
    'prepare_data',
    'train_linear_regression', 
    'train_random_forest'
]