import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path, sep=';', index_col='id', na_values='#N/D')


df = load_data(
    r'C:\Users\ThisFacu\SetUp\Business-Analytics\DataScience-empleados\ProblemaEmpleados\empleados.csv')
df.isna().sum().sort_values(ascending=False)
# 6 targest con valores null.
# 2 targest voy a borrar demaciados valores null.


def graficos_eda_categoricos(cat):
    # Calculamos el numero defilas quenecesitamos
    from math import ceil
    filas = ceil(cat.shape[1] / 2)

    # Definimos el grafico
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas * 6))
    # aplanamos para iterar por el grafico como si fuera 1 dimension en lugar de 2
    ax = ax.flat

    # creamos buvle para añadir al grafico
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=6, fontweight="bold")
        ax[cada].tick_params(labelsize=6)


def estadisticos_eda_cont(num):
    # Calculamos describe
    estadisticos = num.describe().T
    print(estadisticos)
    # Añadimos a la median
    estadisticos['median'] = num.median()
    # Reordeno
    estadisticos = estadisticos.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
    return estadisticos


def explorer_data_(df):
    # Histograma y boxplot para salario_mes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['salario_mes'])
    plt.title('Histograma de salario_mes')
    plt.subplot(1, 2, 2)
    plt.boxplot(df['salario_mes'])
    plt.title('Boxplot de salario_mes')
    plt.show()

    # Gráfico de barras para educacion
    plt.figure(figsize=(8, 5))
    df['educacion'].value_counts().plot(kind='bar')
    plt.title('Distribución de educación')
    plt.xlabel('Nivel educativo')
    plt.ylabel('Frecuencia')
    plt.show()

    # Matriz de correlación
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(8, 6))
    plt.matshow(correlation_matrix)
    plt.title('Matriz de correlación')
    plt.colorbar()
    plt.show()
    correlation_matrix.info()

    # Gráfico de dispersión para edad y salario_mes
    plt.figure(figsize=(8, 6))
    plt.scatter(df['edad'], df['salario_mes'])
    plt.title('Gráfico de dispersión: Edad vs. Salario')
    plt.xlabel('Edad')
    plt.ylabel('Salario mensual')
    plt.show()

    # Gráfico de barras apiladas para departamento y abandono
    pd.crosstab(df['departamento'], df['abandono']
                ).plot(kind='bar', stacked=True)
    plt.title('Distribución de abandono por departamento')
    plt.xlabel('Departamento')
    plt.ylabel('Frecuencia')
    plt.show()

    # Abandonos
    df.abandono.value_counts(normalize=True) * 100
    return df


def clean_data(df):
    df.drop(columns=['anos_en_puesto', 'conciliacion',
            'mayor_edad', 'empleados', 'sexo', 'horas_quincena'], inplace=True)
    return df


def imputar(df):
    df['educacion'] = df['educacion'].fillna('Universitaria')
    df['satisfaccion_trabajo'] = df['satisfaccion_trabajo'].fillna('Alta')
    df['implicacion'] = df['implicacion'].fillna('Alta')
    df['abandono'] = df.abandono.map({'No': 0, 'Yes': 1})


# Mapa de calor para visualizar los valores faltantes


def valorfaltante(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Valores faltantes en el DataFrame')
    plt.show()


# Generacion de insights
# ¿Cual es la taza de abandono?
# rota un 16% del personal de la empreza


# Cuantificacion el problema para la comparacion,
# resumen de info, indentificar patrones
# para los modelos de ML y toma de decisiones

# ¿Hay un perfil de empleado que deja la empreza?


clean_data(df)
imputar(df)
graficos_eda_categoricos(df.select_dtypes('O'))
estadisticos_eda_cont(df.select_dtypes('number'))

explorer_data_(df)
