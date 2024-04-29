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
#Generacion de insights

# Cuantificacion el problema para la comparacion, resumen de info, indentificar patrones
# para los modelos de ML y toma de decisiones
# ¿Cual es la taza de abandono?

df.abandono.value_counts(normalize=True)*100
# nos da el porcentaje sore el total  
# El *100 simplemente se utiliza para convertir el resultado de porcentajes fraccionarios

#rota un 16% del personal de la empreza

#¿Hay un perfil de empleado que deja la empreza?

df['abandono'] = df.abandono.map({'No':0, 'Yes':1})
#   Convertimos a numerica
# Analisis por penetracion
# Educacion
temp = df.groupby('educacion').abandono.mean().sort_values(ascending=False)*100
# la media de una vartiable 0 1 es igual que el porcentaje
temp.plot.bar()
# la taza de abandono de nivel primaria es del 32%
# la taza de abandono de nivel master es del 8%

temp = df.groupby('estado_civil').abandono.mean().sort_values(ascending=False)*100
temp.plot.bar()
# los casados y divorciados se mantienen en el mismo rango 10% - 12%
# los que estan solos es del 25%

temp = df.groupby('horas_extra').abandono.mean().sort_values(ascending=False)*100
temp.plot.bar()
#los empleados con horas extra tienen mas tendencia a dejar el oficio

temp = df.groupby('puesto').abandono.mean().sort_values(ascending=False)*100
temp.plot.bar()
#el 40% de las SalesRepresentative abandono el oficio

temp = df.groupby('abandono').salario_mes.mean()
temp.plot.bar()

#los empleados con sueldo menores tiende a salir de la empreza
    
# Concluciones 
# - El perfil del que deja la empresa
# - Bajo nivel educativo
# - Soltero
# - Trabaja en ventas
# - Bajo salario
# - Alta carga de horas extras
# Estas concluciones y datos de los empleados le podra servir alos de Recursos humanos


# ¿Cual es el impacto económico de este problema?
# el estudio 'Cost of Turnover" del Center for American Progress:

# El coste de la fuga de los empleados que ganan menos de es del 16.1 % de su salario
# El coste de la fuga de los empleados que ganan entre 30000-50000 es del 19,7% de su salario
# El coste de la fuga de los empleados que ganan entre es del de su salario
# El coste de la fuga de los empleados que ganan más de 75000 es del 21 de su salario

#Crearemos la variable anual ya que la que tenemos es mensual
df['salario_ano'] = df.salario_mes.transform(lambda x: x*12 )
df[['salario_mes', 'salario_ano']]

condiciones = [(df['salario_ano'] <= 30000), 
               (df['salario_ano'] > 30000) & (df['salario_ano'] <=50000),
               (df['salario_ano'] > 50000) & (df['salario_ano'] <=75000),
               (df['salario_ano'] > 75000)]

#Lista de resultados.
resultados = [df.salario_ano*0.161, df.salario_ano*0.197, df.salario_ano*0.204, df.salario_ano*0.21]
df['impacto_abandono'] = np.select(condiciones  , resultados, default=-999)
df

#cuanto costo el problema este año
coste_total = df.loc[df.abandono == 1].impacto_abandono.sum()
coste_total

#empleados no motivados cuestan:
df.loc[(df.abandono==1) & (df.implicacion=='Baja')].impacto_abandono.sum()


#cuntanto dinero se podria ahorrar fidelizando a nuestro empleados
print("Reducir un 10% la fuga de empleados nos ahorra",
      (coste_total*0.1), "$ cada año")
print("Reducir un 20% la fuga de empleados nos ahorra",
      (coste_total*0.2), "$ cada año")
print("Reducir un 30% la fuga de empleados nos ahorra",
      (coste_total*0.3), "$ cada año")

#Habíamos Visto que los representantes de ventas son el puesto que más se van. ¿Tendría sentido hacer un plan para ellos? ¿Cual
#sería el coste ahorrado si disminuimos la fuga un 30%?
#Primero vamos a calcular el % de representantes de ventas que se ha ido el año pasado

total_repre_pasado = len(df.loc[df.puesto == 'Sales Representative'])
abandonos_repre_pasado = len(df.loc[(df.puesto == 'Sales Representative') & (df.abandono==1)])
porc_pasado = abandonos_repre_pasado/total_repre_pasado 
porc_pasado
    #el 40% de Sales Representative se fue de la empreza
# estimacion este año

total_repre_actual = len(df.loc[(df.puesto=='Sales Representative') & (df.abandono == 0)])
se_ira = int(total_repre_actual * porc_pasado)
# 19 empleados posiblemente se valla de la empreza  

#Sobre ellos cuantos podemos retener (hipótesis 30%) y cuanto dinero puede suponer
retenemos = int(se_ira * 0.3)
ahorramos = df.loc[(df.puesto=='Sales Representative') & (df.abandono==0),'impacto_abandono'].sum()*0.3

print('Podemos retener', retenemos, 'representantes de ventas y ello supondra ahorrar', ahorramos, '$')

# Este dato también es muy interesante ;nrque nos permite determinar el presupuesto para acciones de retención por departamento o perfil.
# Ya que sabemos que podemos gastarnos hasta 37 sólo en acciones especificas para retener a representantes de ventas y se estarían
# pagando sólas con la pérdida evitada