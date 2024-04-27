# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga de datos
df = pd.read_csv('C:\\Users\\ThisFacu\\PycharmProjects\\DataScience-empleados\\ProblemaEmpleados\\empleados.csv',
                 sep=';', index_col='id', na_values='#N/D')

# Business analytics
df
# La variable Abondono es nuestra variable target


# ----------------Analisis de Calidad De Datos-------------
# Analisis de Nulos
df.isna().sum().sort_values(ascending=False)
# Tenemos 6 variables con valores null
# toma de decicion.
# " Mi desicion fue eliminar las vaiables conciliacion y años_en_puesto
#   ya que tenian demaciados valores null, A las variables sexo, educacion,
#   satisfaccion_trabajo y implicacion seran imputadas."

# Eliminar varaiables.
df.drop(columns=['anos_en_puesto', 'conciliacion'], inplace=True)

df.info()

# EDA: Analisis exploratorios de datos

# EDA Vatiables Categoricas


def graficos_eda_categoricos(cat):
    # Calculamos el numero defilas quenecesitamos
    from math import ceil
    filas = ceil(cat.shape[1]/2)

    # Definimos el grafico
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas*6))
    # aplanamos para iterar por el grafico como si fuera 1 dimension en lugar de 2
    ax = ax.flat

    # creamos buvle para añadir al grafico
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=6, fontweight="bold")
        ax[cada].tick_params(labelsize=6)


graficos_eda_categoricos(df.select_dtypes('O'))
# Visto en grafica
# mas de 230 empleados renunciaron hay aun 1200
# mayor de edad solo tiene un dato asi que sera eliminada
# ImputarVariablesCategoricas
# Educacion:univercitaria, satisfaccion_trabajo:alta, implicacion:alta.

df.drop(columns='mayor_edad', inplace=True)

df['educacion'] = df['educacion'].fillna('Universitaria')

df['satisfaccion_trabajo'] = df['satisfaccion_trabajo'].fillna('Alta')

df['implicacion'] = df['implicacion'].fillna('Alta')

# EDA Variables Numericas


def estadisticos_cont(num):
    # Calculamos describe
    estadisticos = num.describe().T
    print(estadisticos)
    # Añadimos a la median
    estadisticos['median'] = num.median()
    # Reordeno
    estadisticos = estadisticos.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
    return estadisticos


estadisticos_cont(df.select_dtypes('number'))
# Visto en grafica 
# Empleado tiene un valor:Eliminar
# sexo tiene 4 valores, a cortos rasgos es raro que tenga mas 2 de sexos aqsi que seran eliminados
# horas un solo valor lo borro
df.drop(columns=['empleados', 'sexo', 'horas_quincena'], inplace=True)

df.info()
#guardo las columnas
dfcol = df.columns.copy()


df.to_csv('datanew.csv', columns=dfcol, index=False)
