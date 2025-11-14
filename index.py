from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime, timedelta
from meteostat import Point, Daily
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
import io
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import render_template, session, redirect, url_for
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_segura'

# Usuarios de prueba
usuarios = {
    "admin": "agricola1234"
}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        usuario = request.form['usuario']
        clave = request.form['clave']
        if usuario in usuarios and usuarios[usuario] == clave:
            session['usuario'] = usuario
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Usuario o contraseña incorrectos')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'usuario' in session:
        return render_template('dashboard.html', usuario=session['usuario'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('usuario', None)
    return redirect(url_for('login'))

# ----------------------------
# ----------------------------
# Produccion    PANTALLA arriba 1
# ----------------------------
@app.route('/produccion', methods=['GET', 'POST'])
def produccion():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    try:
        # Leer el archivo Excel
        df = pd.read_excel('produccion.xlsx')

        # Normalizar nombres de columnas
        df.columns = df.columns.str.strip().str.upper()

        # Normalizar texto en columnas clave
        for col in ['MES', 'COLOR', 'TIPO', 'VARIEDAD']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        # Asegurar que TALLOS sea numérico
        df['TALLOS'] = pd.to_numeric(df['TALLOS'], errors='coerce').fillna(0)

        # Mapear meses a números
        meses_map = {
            'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
            'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
        }
        df['MES_NUM'] = df['MES'].map(meses_map)

        # Validar columnas necesarias
        columnas_requeridas = {'AÑO', 'MES', 'SEMANA', 'COLOR', 'TIPO', 'VARIEDAD', 'TALLOS'}
        if not columnas_requeridas.issubset(df.columns):
            return "El archivo no contiene todas las columnas necesarias."

        # Limpiar y ordenar datos
        df['SEMANA'] = pd.to_numeric(df['SEMANA'], errors='coerce')
        df.dropna(subset=['SEMANA'], inplace=True) # Eliminar filas donde SEMANA sea nulo
        df = df[(df['SEMANA'] >= 1) & (df['SEMANA'] <= 52)]
        df['SEMANA'] = df['SEMANA'].astype(int)
        df = df.sort_values(by=['AÑO', 'MES_NUM', 'SEMANA'])

        # Tabla HTML para mostrar debajo (opcional)
        tabla_html = df.to_html(classes='table table-striped table-bordered', index=False, border=0)

        # ----- Gráficas -----
        variedades = sorted(df['VARIEDAD'].unique())
        graficas = {}

        for variedad in variedades:
            datos_var = df[df['VARIEDAD'] == variedad]

            if datos_var.empty:
                continue

            # Agrupar por año y semana
            resumen = datos_var.groupby(['AÑO', 'SEMANA'], as_index=False)['TALLOS'].sum()
            if resumen.empty:
                continue

            fig = go.Figure()

            # Añadir líneas por año
            for anio in sorted(resumen['AÑO'].unique()):
                datos_anio = resumen[resumen['AÑO'] == anio].sort_values('SEMANA')
                fig.add_trace(go.Scatter(
                    x=datos_anio['SEMANA'],
                    y=datos_anio['TALLOS'],
                    mode='lines+markers',
                    name=f"Año {anio}"
                ))

            # Configuración estética
            fig.update_layout(
                title=f"Producción semanal - {variedad}",
                xaxis_title='Semana',
                yaxis_title='Tallos',
                template='plotly_white',
                hovermode='x unified',
                xaxis=dict(
                    tickmode='linear',
                    tick0=1,
                    dtick=1,
                    range=[1, 52]
                ),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            
            graficas[variedad] = plot(fig, output_type='div', include_plotlyjs=True)

        return render_template('produccion.html', data=tabla_html, graficas=graficas)

    except Exception as e:
        return f"Error al procesar la producción: {e}"
    
# ----------------------------
# Producción - Gráfica semanal interactiva   PANTALLA abajo 1
# ----------------------------
@app.route('/produccion_dos', methods=['GET', 'POST'])
def produccion_dos():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    # === 1. Leer Excel principal y preparar datos para ENTRENAMIENTO ===
    df = pd.read_excel("produccion.xlsx")
    
    # Crear columna FECHA
    df["FECHA"] = pd.to_datetime(
        df["AÑO"].astype(str) + df["SEMANA"].astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )

    # --- CORRECCIÓN: Identificar el último año COMPLETO para entrenar ---
    año_actual_incompleto = df["AÑO"].max()
    año_max_entrenamiento = año_actual_incompleto - 1

    df_entrenamiento = df[df["AÑO"] <= año_max_entrenamiento]

    df_entrenamiento = df_entrenamiento[df_entrenamiento["AÑO"] >= año_max_entrenamiento - 4]

    # === 1b. Leer Excel de datos reales del año a predecir ===
    df_reales = pd.read_excel("datos_reales.xlsx")
    df_reales["FECHA"] = pd.to_datetime(
        df_reales["AÑO"].astype(str) + df_reales["SEMANA"].astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )
    df_reales = df_reales[df_reales["AÑO"] == año_actual_incompleto]

    # === 2. Generar predicciones por variedad y por semana ===
    filas = []

    for variedad in df_entrenamiento["VARIEDAD"].unique():
        for semana in range(1, 53):
            df_sem = df_entrenamiento[(df_entrenamiento["VARIEDAD"] == variedad) & (df_entrenamiento["SEMANA"] == semana)]
            serie_full = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            serie_hw = serie_full[serie_full.index >= año_max_entrenamiento - 1]
            serie_lr = serie_full[serie_full.index == año_max_entrenamiento]
            serie_lstm = serie_full[serie_full.index >= año_max_entrenamiento - 2]

            if len(serie_full) < 3:
                hw = lr = lstm = 0
            else:
                # --- Promedio simple (2 años) ---
                try:
                    if len(serie_hw) > 0:
                        hw = round(serie_hw.mean(), 2)
                    else:
                        hw = 0
                except:
                    hw = 0

                # --- Gradient Boosting Regressor (1 año) ---
                try:
                    if len(serie_lr) > 1:
                        X = serie_lr.index.values.reshape(-1, 1)
                        y = serie_lr.values

                        model_gb = GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.1,
                            max_depth=3,
                            random_state=42
                        )
                        model_gb.fit(X, y)

                        X_future = np.array([[año_actual_incompleto]])
                        y_future = model_gb.predict(X_future)
                        lr = round(y_future[0], 2)
                    else:
                        lr = round(serie_lr.mean(), 2) if len(serie_lr) > 0 else 0
                except:
                    lr = round(serie_lr.mean(), 2) if len(serie_lr) > 0 else 0

                # === 🔥 LSTM REEMPLAZADO POR PROMEDIO ===
                try:
                    if len(serie_lstm) > 0:
                        lstm = round(serie_lstm.mean(), 2)
                    else:
                        lstm = 0
                except:
                    lstm = 0

            # === Valor real ===
            real_data = df_reales[(df_reales["VARIEDAD"] == variedad) & (df_reales["SEMANA"] == semana)]
            
            if not real_data.empty:
                real = real_data["TALLOS"].sum()
            else:
                real = 0

            def error_pct(real, pred):
                if real == 0:
                    return 0 if pred == 0 else 100.0
                return round(abs(real - pred) / real * 100, 2)

            dif_hw, dif_lr, dif_lstm = real - hw, real - lr, real - lstm
            acc_hw, acc_lr, acc_lstm = error_pct(real, hw), error_pct(real, lr), error_pct(real, lstm)

            filas.append([variedad, semana, hw, lr, lstm, real,
                          dif_hw, dif_lr, dif_lstm, acc_hw, acc_lr, acc_lstm])

    # === 3. DataFrame final ===
    df_pred = pd.DataFrame(filas, columns=[
        "VARIEDAD", "SEMANA", "HW", "LR", "LSTM", "REAL",
        "DIF_HW", "DIF_LR", "DIF_LSTM", "ACC_HW", "ACC_LR", "ACC_LSTM"
    ])

    # === Tabla 1: semana a semana ===
    tabla_semanal = {}
    for variedad in df_pred["VARIEDAD"].unique():
        tabla_semanal[variedad] = df_pred[df_pred["VARIEDAD"] == variedad] \
            .drop(columns="VARIEDAD") \
            .to_dict(orient="records")

    # === Tabla 2: sumas cada 4 semanas por variedad ===
    df_pred["BLOQUE"] = ((df_pred["SEMANA"] - 1) // 4) + 1
    tabla_variedad_df = df_pred.groupby(["VARIEDAD", "BLOQUE"])[
        ["HW", "LR", "LSTM", "REAL"]
    ].sum().reset_index()

    def error_pct(real, pred):
        if real == 0:
            return 0 if pred == 0 else 100.0
        return round(abs(real - pred) / real * 100, 2)

    tabla_variedad_df["DIF_HW"] = tabla_variedad_df["REAL"] - tabla_variedad_df["HW"]
    tabla_variedad_df["DIF_LR"] = tabla_variedad_df["REAL"] - tabla_variedad_df["LR"]
    tabla_variedad_df["DIF_LSTM"] = tabla_variedad_df["REAL"] - tabla_variedad_df["LSTM"]
    tabla_variedad_df["ACC_HW"] = tabla_variedad_df.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_variedad_df["ACC_LR"] = tabla_variedad_df.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_variedad_df["ACC_LSTM"] = tabla_variedad_df.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    tabla_variedad = {}
    for variedad in tabla_variedad_df["VARIEDAD"].unique():
        tabla_variedad[variedad] = tabla_variedad_df[tabla_variedad_df["VARIEDAD"] == variedad] \
            .drop(columns="VARIEDAD") \
            .to_dict(orient="records")

    # === Tabla 3: total anual por variedad ===
    tabla_total = df_pred.groupby("VARIEDAD")[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_total["DIF_HW"] = tabla_total["REAL"] - tabla_total["HW"]
    tabla_total["DIF_LR"] = tabla_total["REAL"] - tabla_total["LR"]
    tabla_total["DIF_LSTM"] = tabla_total["REAL"] - tabla_total["LSTM"]
    tabla_total["ACC_HW"] = tabla_total.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_total["ACC_LR"] = tabla_total.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_total["ACC_LSTM"] = tabla_total.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    # === Tabla 4: total por tipo ===
    df_excel = pd.read_excel("produccion.xlsx")[["VARIEDAD", "TIPO", "COLOR"]].drop_duplicates()
    df_merge = df_pred.merge(df_excel, on="VARIEDAD", how="left")
    
    tabla_tipo = df_merge.groupby("TIPO")[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_tipo["DIF_HW"] = tabla_tipo["REAL"] - tabla_tipo["HW"]
    tabla_tipo["DIF_LR"] = tabla_tipo["REAL"] - tabla_tipo["LR"]
    tabla_tipo["DIF_LSTM"] = tabla_tipo["REAL"] - tabla_tipo["LSTM"]
    tabla_tipo["ACC_HW"] = tabla_tipo.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_tipo["ACC_LR"] = tabla_tipo.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_tipo["ACC_LSTM"] = tabla_tipo.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)
    tabla_tipo = tabla_tipo[tabla_tipo["TIPO"].isin(["COLORES", "ROJO"])]

    # === TABLA 5: general semana a semana ===
    tabla_general_semanal = df_pred.groupby("SEMANA")[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_general_semanal["DIF_HW"] = tabla_general_semanal["REAL"] - tabla_general_semanal["HW"]
    tabla_general_semanal["DIF_LR"] = tabla_general_semanal["REAL"] - tabla_general_semanal["LR"]
    tabla_general_semanal["DIF_LSTM"] = tabla_general_semanal["REAL"] - tabla_general_semanal["LSTM"]
    tabla_general_semanal["ACC_HW"] = tabla_general_semanal.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_general_semanal["ACC_LR"] = tabla_general_semanal.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_general_semanal["ACC_LSTM"] = tabla_general_semanal.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    # === TABLA 6: general por bloques ===
    tabla_general_bloques = df_pred.groupby("BLOQUE")[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_general_bloques["DIF_HW"] = tabla_general_bloques["REAL"] - tabla_general_bloques["HW"]
    tabla_general_bloques["DIF_LR"] = tabla_general_bloques["REAL"] - tabla_general_bloques["LR"]
    tabla_general_bloques["DIF_LSTM"] = tabla_general_bloques["REAL"] - tabla_general_bloques["LSTM"]
    tabla_general_bloques["ACC_HW"] = tabla_general_bloques.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_general_bloques["ACC_LR"] = tabla_general_bloques.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_general_bloques["ACC_LSTM"] = tabla_general_bloques.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    # === TABLA 7: solo COLORES ===
    df_colores = df_merge[df_merge["TIPO"] == "COLORES"]
    tabla_general_colores = df_colores.groupby("BLOQUE")[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_general_colores["DIF_HW"] = tabla_general_colores["REAL"] - tabla_general_colores["HW"]
    tabla_general_colores["DIF_LR"] = tabla_general_colores["REAL"] - tabla_general_colores["LR"]
    tabla_general_colores["DIF_LSTM"] = tabla_general_colores["REAL"] - tabla_general_colores["LSTM"]
    tabla_general_colores["ACC_HW"] = tabla_general_colores.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_general_colores["ACC_LR"] = tabla_general_colores.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_general_colores["ACC_LSTM"] = tabla_general_colores.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    # === TABLA 8: general total ===
    tabla_general_total = pd.DataFrame([{
        "HW": df_pred["HW"].sum(), "LR": df_pred["LR"].sum(),
        "LSTM": df_pred["LSTM"].sum(), "REAL": df_pred["REAL"].sum()
    }])
    tabla_general_total["DIF_HW"] = tabla_general_total["REAL"] - tabla_general_total["HW"]
    tabla_general_total["DIF_LR"] = tabla_general_total["REAL"] - tabla_general_total["LR"]
    tabla_general_total["DIF_LSTM"] = tabla_general_total["REAL"] - tabla_general_total["LSTM"]
    tabla_general_total["ACC_HW"] = tabla_general_total.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_general_total["ACC_LR"] = tabla_general_total.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_general_total["ACC_LSTM"] = tabla_general_total.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    # === TABLA 9: color total ===
    tabla_color_total = df_merge.groupby("COLOR")[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_color_total["DIF_HW"] = tabla_color_total["REAL"] - tabla_color_total["HW"]
    tabla_color_total["DIF_LR"] = tabla_color_total["REAL"] - tabla_color_total["LR"]
    tabla_color_total["DIF_LSTM"] = tabla_color_total["REAL"] - tabla_color_total["LSTM"]
    tabla_color_total["ACC_HW"] = tabla_color_total.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_color_total["ACC_LR"] = tabla_color_total.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_color_total["ACC_LSTM"] = tabla_color_total.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    # === TABLA 10: color por bloques ===
    tabla_color_bloques = df_merge.groupby(["COLOR", "BLOQUE"])[["HW", "LR", "LSTM", "REAL"]].sum().reset_index()
    tabla_color_bloques["DIF_HW"] = tabla_color_bloques["REAL"] - tabla_color_bloques["HW"]
    tabla_color_bloques["DIF_LR"] = tabla_color_bloques["REAL"] - tabla_color_bloques["LR"]
    tabla_color_bloques["DIF_LSTM"] = tabla_color_bloques["REAL"] - tabla_color_bloques["LSTM"]
    tabla_color_bloques["ACC_HW"] = tabla_color_bloques.apply(lambda x: error_pct(x["REAL"], x["HW"]), axis=1)
    tabla_color_bloques["ACC_LR"] = tabla_color_bloques.apply(lambda x: error_pct(x["REAL"], x["LR"]), axis=1)
    tabla_color_bloques["ACC_LSTM"] = tabla_color_bloques.apply(lambda x: error_pct(x["REAL"], x["LSTM"]), axis=1)

    def get_color(value):
        if value <= 20: return "#d4edda"
        elif 21 <= value <= 30: return "#fff3cd"
        elif 31 <= value <= 50: return "#f8d7da"
        else: return "#f5c6cb"

    try:
        max_semana = int(df_reales["SEMANA"].max())
        if pd.isna(max_semana) or max_semana < 1: max_semana = 52
    except Exception:
        max_semana = 52

    datos_interactivos = {
        "tabla_total": tabla_total.to_dict(orient="records"),
        "tabla_tipo": tabla_tipo.to_dict(orient="records"),
        "tabla_general_semanal": tabla_general_semanal.to_dict(orient="records"),
        "tabla_general_bloques": tabla_general_bloques.to_dict(orient="records"),
        "tabla_general_colores": tabla_general_colores.to_dict(orient="records"),
        "tabla_general_total": tabla_general_total.to_dict(orient="records"),
        "tabla_color_total": tabla_color_total.to_dict(orient="records"),
        "tabla_color_bloques": tabla_color_bloques.to_dict(orient="records")
    }

    return render_template(
        "produccion_dos.html",
        tabla_semanal=tabla_semanal,
        tabla_variedad=tabla_variedad,
        tabla_total=tabla_total.to_dict(orient="records"),
        tabla_tipo=tabla_tipo.to_dict(orient="records"),
        tabla_general_semanal=tabla_general_semanal.to_dict(orient="records"),
        tabla_general_bloques=tabla_general_bloques.to_dict(orient="records"),
        tabla_general_colores=tabla_general_colores.to_dict(orient="records"),
        tabla_general_total=tabla_general_total.to_dict(orient="records"),
        tabla_color_total=tabla_color_total.to_dict(orient="records"),
        tabla_color_bloques=tabla_color_bloques.to_dict(orient="records"),
        get_color=get_color,
        datos_interactivos=datos_interactivos,
        max_semana=max_semana
    )

# ----------------------------
# Prediccion - Tabla      PANTALLA 2 arriba
# ----------------------------
@app.route('/prediccion_tabla')
def prediccion_tabla():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    # === 1. Leer Excel principal y no filtrarlo todavía ===
    df_original = pd.read_excel("produccion.xlsx")
    # Columnas esperadas: AÑO | MES | SEMANA | COLOR | TIPO | VARIEDAD | TALLOS

    # Crear columna FECHA en el dataframe original
    df_original["FECHA"] = pd.to_datetime(
        df_original["AÑO"].astype(str) + df_original["SEMANA"].astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )

    # Determinar última semana real antes de filtrar
    ultimo_año_real = df_original["AÑO"].max()
    ultima_semana_real = df_original[df_original["AÑO"] == ultimo_año_real]["SEMANA"].max()

    # === 1b. Crear un DataFrame HISTÓRICO solo con años completos para entrenar ===
    semanas_por_año = df_original.groupby("AÑO")["SEMANA"].nunique()
    años_completos = semanas_por_año[semanas_por_año >= 52].index
    df_historico = df_original[df_original["AÑO"].isin(años_completos)].copy()

    # Filtrar últimos 5 años completos en el histórico
    año_max_historico = df_historico["AÑO"].max()
    df_historico = df_historico[df_historico["AÑO"] >= año_max_historico - 5]

    # === 2. Generar predicciones ===
    filas = []

    # Iterar sobre las variedades del histórico
    for variedad in df_historico["VARIEDAD"].unique():

        semana_actual_pred = ultima_semana_real + 1
        año_actual_pred = ultimo_año_real
        semanas_a_generar = 8

        for i in range(semanas_a_generar):

            semana_a_predecir = semana_actual_pred
            año_a_predecir = año_actual_pred

            # Si la semana supera 52, pasa al siguiente año
            if semana_a_predecir > 52:
                semana_a_predecir -= 52
                año_a_predecir += 1

            # Preparación de datos para modelos
            df_sem = df_historico[
                (df_historico["VARIEDAD"] == variedad) &
                (df_historico["SEMANA"] == semana_a_predecir)
            ]

            serie_full = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            serie_hw = serie_full[serie_full.index >= año_max_historico - 1]
            serie_lstm = serie_full[serie_full.index >= año_max_historico - 2]

            if len(serie_full) < 3:
                hw = 0
                lstm = 0
            else:
                # Promedio simple (2 años)
                try:
                    hw = round(serie_hw.mean(), 2) if len(serie_hw) > 0 else 0
                except:
                    hw = 0

                #  SE ELIMINA COMPLETAMENTE EL GRADIENT BOOSTING
                # lr = None    # ya no se usa

                # Reemplazo LSTM: promedio simple usando 3 años
                try:
                    lstm = round(serie_lstm.mean(), 2) if len(serie_lstm) > 0 else 0
                except:
                    lstm = 0

            #  SE ELIMINA TODO LO RELACIONADO CON DATOS REALES, DIFERENCIAS Y ACCURACY

            filas.append([
                variedad, año_a_predecir, semana_a_predecir, hw, lstm
            ])

            semana_actual_pred += 1

    # === TABLA 1: semana a semana por variedad ===
    df_pred = pd.DataFrame(filas, columns=[
        "VARIEDAD", "AÑO", "SEMANA", "HW", "LSTM"
    ])

    tabla_semanal = {}
    for variedad in df_pred["VARIEDAD"].unique():
        tabla_semanal[variedad] = df_pred[df_pred["VARIEDAD"] == variedad] \
            .drop(columns="VARIEDAD") \
            .to_dict(orient="records")

    # ORDENAR LAS TABLAS ALFABÉTICAMENTE POR NOMBRE DE VARIEDAD
    tabla_semanal = dict(sorted(tabla_semanal.items()))

    def get_color(value):
        return "#ffffff"

    return render_template(
        "prediccion_tabla.html",
        tabla_semanal=tabla_semanal,
        get_color=get_color,
        max_semana=ultima_semana_real
    )


# ----------------------------
# Prediccion - Tabla_dos     pantalla abajo 2  
# ----------------------------
@app.route('/prediccion_tabla_dos')
def prediccion_tabla_dos():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    # === 1. Leer Excel principal ===
    df_original = pd.read_excel("produccion.xlsx")
    df_original["FECHA"] = pd.to_datetime(
        df_original["AÑO"].astype(str) + df_original["SEMANA"].astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )

    # === 1b. Crear DataFrame HISTÓRICO con años completos ===
    semanas_por_año = df_original.groupby("AÑO")["SEMANA"].nunique()
    años_completos = semanas_por_año[semanas_por_año >= 52].index
    df_historico = df_original[df_original["AÑO"].isin(años_completos)].copy()

    año_max_historico = df_historico["AÑO"].max()
    df_historico = df_historico[df_historico["AÑO"] >= año_max_historico - 5]

    # === 2. Generar predicciones solo HW y LSTM ===
    filas = []

    for variedad in df_historico["VARIEDAD"].unique():
        for semana in range(1, 9):

            df_sem = df_historico[(df_historico["VARIEDAD"] == variedad) &
                                  (df_historico["SEMANA"] == semana)]

            serie_full = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            serie_hw = serie_full[serie_full.index >= año_max_historico - 1]
            serie_lstm = serie_full[serie_full.index >= año_max_historico - 2]

            if len(serie_full) < 3:
                hw = lstm = 0
            else:
                # --- Promedio simple (HW) ---
                try:
                    hw = round(serie_hw.mean(), 2) if len(serie_hw) > 0 else 0
                except:
                    hw = 0

                # --- PROMEDIO simple (sustituto de LSTM) ---
                try:
                    lstm = round(serie_lstm.mean(), 2) if len(serie_lstm) > 0 else 0
                except:
                    lstm = 0

            # Guardar solo predicciones
            filas.append([
                variedad, semana, hw, lstm
            ])

    df_pred = pd.DataFrame(filas, columns=[
        "VARIEDAD", "SEMANA", "HW", "LSTM"
    ])

    # === Crear bloques ===
    df_pred["BLOQUE"] = ((df_pred["SEMANA"] - 1) // 4) + 1

    # === Tabla por variedad ===
    tabla_variedad_df = df_pred.groupby(["VARIEDAD", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()

    # *** ORDENAR VARIEDADES ALFABÉTICAMENTE ***
    tabla_variedad = {
        variedad: tabla_variedad_df[tabla_variedad_df["VARIEDAD"] == variedad]
        .drop(columns="VARIEDAD")
        .to_dict(orient="records")
        for variedad in sorted(tabla_variedad_df["VARIEDAD"].unique())
    }

    # === Tabla general semanal ===
    tabla_general_semanal = df_pred.groupby("SEMANA")[["HW", "LSTM"]].sum().reset_index()

    # === Tabla general para COLORES (solo usando produccion.xlsx) ===
    df_excel = pd.read_excel("produccion.xlsx")[["VARIEDAD", "TIPO", "COLOR"]].drop_duplicates()
    df_merge = df_pred.merge(df_excel, on="VARIEDAD", how="left")

    df_colores = df_merge[df_merge["TIPO"] == "COLORES"]

    tabla_general_colores = df_colores.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    # === Función color === (si quieres usarla para pintar algo)
    def get_color(value):
        if value <= 15:
            return "#d4edda"
        elif 16 <= value <= 30:
            return "#fff3cd"
        elif 31 <= value <= 50:
            return "#f8d7da"
        else:
            return "#f5c6cb"

    return render_template(
        "prediccion_tabla_dos.html",
        tabla_variedad=tabla_variedad,
        tabla_general_semanal=tabla_general_semanal.to_dict(orient="records"),
        tabla_general_colores=tabla_general_colores.to_dict(orient="records"),
        get_color=get_color
    )

# ----------------------------
# Prediccion - grafica   PANTALLA 3 arriba
# ----------------------------

@app.route('/prediccion_grafica', methods=['GET', 'POST'])
def prediccion_grafica():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    # Selección de rango de semanas
    limite_semanas = 53
    rango_seleccionado = '1-52'

    if request.method == 'POST':
        rango_seleccionado = request.form.get('rango_semanas')
        if rango_seleccionado == '1-27':
            limite_semanas = 27

    # === 1. Leer Excel principal (único archivo ahora) ===
    try:
        df = pd.read_excel("produccion.xlsx")
    except FileNotFoundError as e:
        return f"<h1>Error</h1><p>No se encontró el archivo: <strong>{e.filename}</strong>.</p>"

    df["FECHA"] = pd.to_datetime(
        df["AÑO"].astype(str) + df["SEMANA"].astype(str) + "1",
        format="%G%V%w",
        errors="coerce"
    )
    año_max = df["AÑO"].max()
    df = df[df["AÑO"] >= año_max - 5]

    # === 2. Generar predicciones por variedad y semana ===
    filas = []

    for variedad in df["VARIEDAD"].unique():
        for semana in range(1, limite_semanas):

            df_sem = df[(df["VARIEDAD"] == variedad) & (df["SEMANA"] == semana)]
            serie_full = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            # Datos recientes
            serie_hw = serie_full[serie_full.index >= año_max - 1]
            serie_lstm = serie_full[serie_full.index >= año_max - 2]

            # Predicciones finales
            if len(serie_full) < 3:
                hw = 0
                lstm = 0
            else:
                try:
                    hw = round(serie_hw.mean(), 2) if len(serie_hw) > 0 else 0
                except:
                    hw = 0

                # LSTM → ahora es solamente un promedio simple
                try:
                    lstm = round(serie_lstm.mean(), 2)
                except:
                    lstm = 0

            filas.append([variedad, semana, hw, lstm])

    # === 3. DataFrame final ===
    df_pred = pd.DataFrame(filas, columns=["VARIEDAD", "SEMANA", "HW", "LSTM"])

    if df_pred.empty:
        return render_template(
            "prediccion_grafica.html",
            error="No se generaron datos.",
            rango_actual=rango_seleccionado
        )

    # ==============================================================================
    # 4. TABLAS (SIN REAL, SIN DIFERENCIAS, SIN PORCENTAJES)
    # ==============================================================================

    # === Tabla 1: semana a semana por variedad ===
    tabla_semanal = {
        v: df_pred[df_pred["VARIEDAD"] == v]
            .drop(columns="VARIEDAD")
            .to_dict(orient="records")
        for v in df_pred["VARIEDAD"].unique()
    }

    # Añadimos columna de bloque
    df_pred["BLOQUE"] = ((df_pred["SEMANA"] - 1) // 4) + 1

    # === Tabla 2: sumas cada 4 semanas por variedad ===
    tabla_variedad_df = df_pred.groupby(["VARIEDAD", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()
    tabla_variedad = {
        v: tabla_variedad_df[tabla_variedad_df["VARIEDAD"] == v]
            .drop(columns="VARIEDAD")
            .to_dict(orient="records")
        for v in tabla_variedad_df["VARIEDAD"].unique()
    }

    # === Tabla 3: total por variedad ===
    tabla_total = df_pred.groupby("VARIEDAD")[["HW", "LSTM"]].sum().reset_index()

    # === Tabla 4: total por tipo ===
    df_excel = pd.read_excel("produccion.xlsx")[["VARIEDAD", "TIPO", "COLOR"]].drop_duplicates()
    df_merge = df_pred.merge(df_excel, on="VARIEDAD", how="left")
    tabla_tipo = df_merge.groupby("TIPO")[["HW", "LSTM"]].sum().reset_index()
    tabla_tipo = tabla_tipo[tabla_tipo["TIPO"].isin(["COLORES", "ROJO"])]

    # === Tabla 5: general semana a semana ===
    tabla_general_semanal = df_pred.groupby("SEMANA")[["HW", "LSTM"]].sum().reset_index()

    # === Tabla 6: general por bloques ===
    tabla_general_bloques = df_pred.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    # === Tabla 7: solo COLORES por bloques ===
    df_colores = df_merge[df_merge["TIPO"] == "COLORES"]
    tabla_general_colores = df_colores.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    # === Tabla 8: total general ===
    tabla_general_total = pd.DataFrame([df_pred[["HW", "LSTM"]].sum().to_dict()])

    # === Tabla 9: total por COLOR ===
    tabla_color_total = df_merge.groupby("COLOR")[["HW", "LSTM"]].sum().reset_index()

    # === Tabla 10: por COLOR en bloques ===
    tabla_color_bloques = df_merge.groupby(["COLOR", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()

    # Color de celdas (opcional)
    def get_color(value):
        return "#FFFFFF"

    # === Determinar max_semana dinámicamente ===
    try:
        max_semana = int(df_pred["SEMANA"].max())
    except:
        max_semana = 0

    # === Render ===
    return render_template(
        "prediccion_grafica.html",
        tabla_semanal=tabla_semanal,
        tabla_variedad=tabla_variedad,
        tabla_total=tabla_total.to_dict(orient="records"),
        tabla_tipo=tabla_tipo.to_dict(orient="records"),
        tabla_general_semanal=tabla_general_semanal.to_dict(orient="records"),
        tabla_general_bloques=tabla_general_bloques.to_dict(orient="records"),
        tabla_general_colores=tabla_general_colores.to_dict(orient="records"),
        tabla_general_total=tabla_general_total.to_dict(orient="records"),
        tabla_color_total=tabla_color_total.to_dict(orient="records"),
        tabla_color_bloques=tabla_color_bloques.to_dict(orient="records"),
        get_color=get_color,
        max_semana=max_semana,
        rango_actual=rango_seleccionado
    )

# ----------------------------
# Resumen - dos    PANTALLA 8
# ----------------------------
@app.route('/resumen_dos')
def resumen_dos():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    limite_semanas = 53  
    rango_seleccionado = '1-53'

    if request.method == 'POST':
        rango_seleccionado = request.form.get('rango_semanas')
        if rango_seleccionado == '1-53':
            limite_semanas = 53

    # ================================
    # 1. CARGA DE ARCHIVO ÚNICO
    # ================================
    try:
        df = pd.read_excel("produccion.xlsx")
    except FileNotFoundError as e:
        return f"<h1>Error</h1><p>No se encontró el archivo: <strong>{e.filename}</strong>.</p>"

    df["FECHA"] = pd.to_datetime(df["AÑO"].astype(str) + df["SEMANA"].astype(str) + "1",
                                 format="%G%V%w", errors="coerce")

    año_max = df["AÑO"].max()
    df = df[df["AÑO"] >= año_max - 5]

    filas = []

    # ================================
    # 2. CÁLCULO DE PREDICCIONES (SOLO HW Y LSTM)
    # ================================
    for variedad in df["VARIEDAD"].unique():

        for semana in range(1, limite_semanas):

            df_sem = df[(df["VARIEDAD"] == variedad) & (df["SEMANA"] == semana)]
            serie_full = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            serie_hw = serie_full[serie_full.index >= año_max - 1]
            serie_lstm = serie_full[serie_full.index >= año_max - 2]

            if len(serie_full) < 2:
                hw = 0
                lstm = 0
            else:
                try:
                    hw = round(serie_hw.mean(), 2) if len(serie_hw) > 0 else 0
                except:
                    hw = 0

                try:
                    lstm = round(serie_lstm.mean(), 2) if len(serie_lstm) > 0 else 0
                except:
                    lstm = 0

            filas.append([variedad, semana, hw, lstm])

    # ================================
    # 3. DATAFRAME FINAL (SIN REAL, SIN LR)
    # ================================
    df_pred = pd.DataFrame(filas, columns=[
        "VARIEDAD", "SEMANA", "HW", "LSTM"
    ])

    if df_pred.empty:
        return render_template("resumen_dos.html",
                               error="No se generaron datos.",
                               rango_actual=rango_seleccionado)

    # ===============================================
    # 4. TABLAS (TODAS FUNCIONAN SOLO CON HW Y LSTM)
    # ===============================================

    # TABLA 1 — Semana a semana por variedad
    tabla_semanal = {
        v: df_pred[df_pred["VARIEDAD"] == v]
            .drop(columns="VARIEDAD")
            .to_dict(orient="records")
        for v in df_pred["VARIEDAD"].unique()
    }

    # BLOQUES DE 4 SEMANAS
    df_pred["BLOQUE"] = ((df_pred["SEMANA"] - 1) // 4) + 1

    # TABLA 2 — Suma por variedad y bloque
    tabla_variedad_df = df_pred.groupby(["VARIEDAD", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()
    tabla_variedad = {
        v: tabla_variedad_df[tabla_variedad_df["VARIEDAD"] == v]
            .drop(columns="VARIEDAD")
            .to_dict(orient="records")
        for v in tabla_variedad_df["VARIEDAD"].unique()
    }

    # TABLA 3 — Total anual por variedad
    tabla_total = df_pred.groupby("VARIEDAD")[["HW", "LSTM"]].sum().reset_index()

    # TABLA 4 — Total por tipo
    df_excel = pd.read_excel("produccion.xlsx")[["VARIEDAD", "TIPO", "COLOR"]].drop_duplicates()
    df_merge = df_pred.merge(df_excel, on="VARIEDAD", how="left")

    tabla_tipo = df_merge.groupby("TIPO")[["HW", "LSTM"]].sum().reset_index()
    tabla_tipo = tabla_tipo[tabla_tipo["TIPO"].isin(["COLORES", "ROJO"])]

    # TABLA 5 — General semana a semana
    tabla_general_semanal = df_pred.groupby("SEMANA")[["HW", "LSTM"]].sum().reset_index()

    # TABLA 6 — General por bloques
    tabla_general_bloques = df_pred.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    # TABLA 7 — Solo colores en bloques
    df_colores = df_merge[df_merge["TIPO"] == "COLORES"]
    tabla_general_colores = df_colores.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    # TABLA 8 — Total general
    tabla_general_total = pd.DataFrame([df_pred[["HW", "LSTM"]].sum().to_dict()])

    # TABLA 9 — Total por color
    tabla_color_total = df_merge.groupby("COLOR")[["HW", "LSTM"]].sum().reset_index()

    # TABLA 10 — Color por bloques
    tabla_color_bloques = df_merge.groupby(["COLOR", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()

    # COLOR DE CELDAS (MANTENIDO POR SI AÚN LO USAS)
    def get_color(value):
        return "#FFFFFF"

    # MÁXIMO SEMANA
    try:
        max_semana = int(df_pred["SEMANA"].max())
    except:
        max_semana = 0

    # DATOS INTERACTIVOS LIMPIOS
    datos_interactivos = {
        "tabla_total": tabla_total.to_dict(orient="records"),
        "tabla_tipo": tabla_tipo.to_dict(orient="records"),
        "tabla_general_semanal": tabla_general_semanal.to_dict(orient="records"),
        "tabla_general_bloques": tabla_general_bloques.to_dict(orient="records"),
        "tabla_general_colores": tabla_general_colores.to_dict(orient="records"),
        "tabla_general_total": tabla_general_total.to_dict(orient="records"),
        "tabla_color_total": tabla_color_total.to_dict(orient="records"),
        "tabla_color_bloques": tabla_color_bloques.to_dict(orient="records")
    }

    # RENDER FINAL
    return render_template(
        "resumen_dos.html",
        tabla_semanal=tabla_semanal,
        tabla_variedad=tabla_variedad,
        tabla_total=tabla_total.to_dict(orient="records"),
        tabla_tipo=tabla_tipo.to_dict(orient="records"),
        tabla_general_semanal=tabla_general_semanal.to_dict(orient="records"),
        tabla_general_bloques=tabla_general_bloques.to_dict(orient="records"),
        tabla_general_colores=tabla_general_colores.to_dict(orient="records"),
        tabla_general_total=tabla_general_total.to_dict(orient="records"),
        tabla_color_total=tabla_color_total.to_dict(orient="records"),
        tabla_color_bloques=tabla_color_bloques.to_dict(orient="records"),
        get_color=get_color,
        datos_interactivos=datos_interactivos,
        max_semana=max_semana,
        rango_actual=rango_seleccionado
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

