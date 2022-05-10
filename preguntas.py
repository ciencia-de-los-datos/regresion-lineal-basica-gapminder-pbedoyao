"""
RegresiÃ³n Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirÃ¡ un modelo de regresiÃ³n lineal univariado.

"""
import numpy as np
import pandas as pd
import sys
import preguntas


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el cÃ³digo presentado a continuaciÃ³n.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('./gm_2008_region.csv')
    
    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df['life'].values
    X = df['fertility'].values
    
    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.reshape(139, 1)

    # Trasforme `X` a un array de numpy usando|| reshape
    X_reshaped = X.reshape(139, 1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresiÃ³n de algunas estadÃ­sticas bÃ¡sicas
    Complete el cÃ³digo presentado a continuaciÃ³n.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('./gm_2008_region.csv')

    # Imprima las dimensiones del DataFrame
    print(df.shape)

    # Imprima la correlaciÃ³n entre las columnas `life` y `fertility` con 4 decimales.
    print(round(df['life'].corr(df['fertility'], method='pearson'),4))
    
    # Imprima la media de la columna `life` con 4 decimales.
    print(round(df["life"].mean(),4))

    # Imprima el tipo de dato de la columna `fertility`.
    print(type(df['fertility']))

    # Imprima la correlaciÃ³n entre las columnas `GDP` y `life` con 4 decimales.
    print(round(df['GDP'].corr(df['life'], method='pearson'),4))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el cÃ³digo presentado a continuaciÃ³n.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('./gm_2008_region.csv')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df['fertility'].values

    # Asigne a la variable los valores de la columna `life`
    y_life = df['life'].values

    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresiÃ³n lineal
    reg = LinearRegression()

    # Cree El espacio de predicciÃ³n. Esto es, use linspace para crear
    # un vector con valores entre el mÃ¡ximo y el mÃ­nimo de X_fertility
    prediction_space = np.linspace(
        min(X_fertility),
        max(X_fertility),
    ).reshape(-1, 1)

    # Entrene el modelo usando X_fertility y y_life
    reg.fit(X_fertility.reshape(-1, 1), y_life.reshape(-1, 1))

    # Compute las predicciones para el espacio de predicciÃ³n
    y_pred = reg.predict(prediction_space.reshape(-1, 1))

    # Imprima el R^2 del modelo con 4 decimales
    print(reg.score(X_fertility.reshape(-1, 1), y_life.reshape(-1, 1)).round(4))


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el cÃ³digo presentado a continuaciÃ³n.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from sklearn.model_selection import train_test_split

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('./gm_2008_region.csv')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df['fertility'].values

    # Asigne a la variable los valores de la columna `life`
    y_life = df['life'].values

    # Divida los datos de entrenamiento y prueba. La semilla del generador de nÃºmeros
    # aleatorios es 53. El tamaÃ±o de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility,
        y_life,
        test_size=0.2,
        random_state=53,
    )
    
    # Cree una instancia del modelo de regresiÃ³n lineal
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    reg.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

    # Pronostique y_test usando X_test
    y_pred = reg.predict(X_test.reshape(-1, 1))

    # Compute and print R^2 and RMSE
    from sklearn.metrics import mean_squared_error
    print("R^2: {:6.4f}".format(reg.score(X_test.reshape(-1, 1), y_test.reshape(-1, 1))))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
