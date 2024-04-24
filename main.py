
# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga de datos
df  = pd.read_csv('./DF/AbandonoEmpleados.csv', sep=';', index_col='id', na_values='#N/D')

print(df)