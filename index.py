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
# PANTALLA ARRIBA 1
# ----------------------------
@app.route('/produccion', methods=['GET'])
def produccion():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    try:
        # Leer Excel
        df = pd.read_excel('produccion.xlsx')

        # Normalizar columnas
        df.columns = df.columns.str.strip().str.upper()
        for col in ['MES', 'COLOR', 'TIPO', 'VARIEDAD']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        df['TALLOS'] = pd.to_numeric(df['TALLOS'], errors='coerce').fillna(0)

        # Crear lista de variedades
        variedades = sorted(df['VARIEDAD'].unique())

        # Obtener variedad seleccionada
        variedad_sel = request.args.get("variedad")

        grafica_html = None

        if variedad_sel:
            datos_var = df[df['VARIEDAD'] == variedad_sel]
            resumen = datos_var.groupby(['AÑO', 'SEMANA'], as_index=False)['TALLOS'].sum()

            if not resumen.empty:
                fig = go.Figure()

                # Líneas por año
                for anio in sorted(resumen['AÑO'].unique()):
                    datos_anio = resumen[resumen['AÑO'] == anio].sort_values('SEMANA')

                    fig.add_trace(go.Scatter(
                        x=datos_anio['SEMANA'],
                        y=datos_anio['TALLOS'],
                        mode='lines+markers',
                        name=f"Año {anio}"
                    ))

                fig.update_layout(
                    title=f"Producción semanal - {variedad_sel}",
                    xaxis_title='Semana',
                    yaxis_title='Tallos',
                    template='plotly_white',
                    hovermode='x unified'
                )

                # Cargar plotly una sola vez
                grafica_html = plot(fig, output_type='div', include_plotlyjs=True)

        # Render
        return render_template(
            'produccion.html',
            variedades=variedades,
            variedad_sel=variedad_sel,
            grafica_html=grafica_html
        )
    except Exception as e:
        return f"Error al procesar la producción: {e}"
   
# ----------------------------
#  PANTALLA ABAJO 1
# ----------------------------
@app.route('/produccion_dos', methods=['GET', 'POST'])
def produccion_dos():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    import pandas as pd
    import math

    # =============================
    # 1. Leer datos históricos
    # =============================
    df = pd.read_excel("produccion.xlsx")

    df["FECHA"] = pd.to_datetime(
        df["AÑO"].astype(str) + df["SEMANA"].astype(str) + "1",
        format="%G%V%w",
        errors="coerce"
    )

    año_actual = df["AÑO"].max()
    año_max_entrenamiento = año_actual - 1

    df_entrenamiento = df[
        (df["AÑO"] <= año_max_entrenamiento) &
        (df["AÑO"] >= año_max_entrenamiento - 3)
    ]

    # =============================
    # 2. Leer datos reales
    # =============================
    df_reales = pd.read_excel("datos_reales.xlsx")

    df_reales["FECHA"] = pd.to_datetime(
        df_reales["AÑO"].astype(str) + df_reales["SEMANA"].astype(str) + "1",
        format="%G%V%w",
        errors="coerce"
    )

    df_reales = df_reales[df_reales["AÑO"] == año_actual]

    # =============================
    # 3. Error %
    # =============================
    def error_pct(real, pred):
        if real is None or pred is None or real == 0:
            return None
        return round(abs(real - pred) / real * 100, 2)

    # =============================
    # 4. Predicción semanal base
    # =============================
    filas = []

    for variedad in sorted(df_entrenamiento["VARIEDAD"].unique()):
        for semana in range(1, 53):

            df_sem = df_entrenamiento[
                (df_entrenamiento["VARIEDAD"] == variedad) &
                (df_entrenamiento["SEMANA"] == semana)
            ]

            serie = (
                df_sem.sort_values("AÑO")
                .set_index("AÑO")["TALLOS"]
                .dropna()
            )

            if len(serie) < 3:
                hw = None
                lstm = None
            else:
                hw = round(serie[serie.index >= año_max_entrenamiento].mean(), 2)
                lstm = round(serie[serie.index >= año_max_entrenamiento - 1].mean(), 2)

            real_data = df_reales[
                (df_reales["VARIEDAD"] == variedad) &
                (df_reales["SEMANA"] == semana)
            ]

            real = real_data["TALLOS"].sum() if not real_data.empty else None

            filas.append([variedad, semana, hw, lstm, real])

    df_base = pd.DataFrame(filas, columns=["VARIEDAD", "SEMANA", "HW", "LSTM", "REAL"])

    # =============================
    # 5. Construir tablas agregadas (ANTI NaN)
    # =============================
    def build_table(df, label):
        tabla = []
        for key, g in df.groupby(label):
            hw = g["HW"].sum(min_count=1)
            lstm = g["LSTM"].sum(min_count=1)
            real = g["REAL"].sum(min_count=1)

            hw = None if pd.isna(hw) else hw
            lstm = None if pd.isna(lstm) else lstm
            real = None if pd.isna(real) else real

            acc_hw = error_pct(real, hw)
            acc_lstm = error_pct(real, lstm)

            tabla.append({
                label: int(key),
                "HW": hw,
                "LSTM": lstm,
                "REAL": real,
                "DIF_HW": real - hw if real is not None and hw is not None else None,
                "DIF_LSTM": real - lstm if real is not None and lstm is not None else None,
                "ACC_HW": acc_hw,
                "ACC_LSTM": acc_lstm
            })
        return tabla

    variedades = sorted(df_base["VARIEDAD"].unique())
    variedad_sel = request.form.get("variedad")

    tabla_12 = tabla_26 = tabla_52 = []

    if variedad_sel:
        df_v = df_base[df_base["VARIEDAD"] == variedad_sel].copy()

        df_v["MES"] = ((df_v["SEMANA"] - 1) // 4) + 1
        tabla_12 = build_table(df_v, "MES")

        df_v["Q"] = ((df_v["SEMANA"] - 1) // 2) + 1
        tabla_26 = build_table(df_v, "Q")

        tabla_52 = build_table(df_v, "SEMANA")

    # =============================
    # 6. Color SOLO con valor válido
    # =============================
    def get_color(val):
        if val is None or isinstance(val, float) and math.isnan(val):
            return "transparent"
        if val <= 25:
            return "#bbf7d0"
        elif val <= 30:
            return "#fde68a"
        elif val <= 50:
            return "#fecaca"
        return "#fca5a5"

    return render_template(
        "produccion_dos.html",
        variedades=variedades,
        variedad_sel=variedad_sel,
        tabla_12=tabla_12,
        tabla_26=tabla_26,
        tabla_52=tabla_52,
        get_color=get_color
    )


# ----------------------------
# PANTALLA 2 ARRIBA
# ----------------------------
@app.route('/prediccion_tabla')
def prediccion_tabla():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    import pandas as pd
    from flask import request, render_template

    # === 1. Leer Excel ===
    df_original = pd.read_excel("produccion.xlsx")

    df_original.columns = df_original.columns.str.strip().str.upper()
    df_original["VARIEDAD"] = df_original["VARIEDAD"].astype(str).str.strip().str.upper()

    # Última semana real
    ultimo_año_real = df_original["AÑO"].max()
    ultima_semana_real = (
        df_original[df_original["AÑO"] == ultimo_año_real]["SEMANA"].max()
    )

    # === 2. Histórico (años completos) ===
    semanas_por_año = df_original.groupby("AÑO")["SEMANA"].nunique()
    años_completos = semanas_por_año[semanas_por_año >= 52].index
    df_historico = df_original[df_original["AÑO"].isin(años_completos)].copy()

    año_max_historico = df_historico["AÑO"].max()
    df_historico = df_historico[df_historico["AÑO"] >= año_max_historico - 5]

    # === 3. Predicciones semanales (8 semanas) ===
    filas = []

    for variedad in df_historico["VARIEDAD"].unique():

        semana_actual = ultima_semana_real + 1
        año_actual = ultimo_año_real

        for _ in range(8):

            semana = semana_actual
            año = año_actual

            if semana > 52:
                semana -= 52
                año += 1

            df_sem = df_historico[
                (df_historico["VARIEDAD"] == variedad) &
                (df_historico["SEMANA"] == semana)
            ]

            serie = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            hw = serie.tail(2).mean() if len(serie) >= 2 else 0
            lstm = serie.tail(3).mean() if len(serie) >= 3 else 0

            filas.append([variedad, hw, lstm])
            semana_actual += 1

    df_pred = pd.DataFrame(filas, columns=["VARIEDAD", "HW", "LSTM"])

    # === 4. Agrupar por MES (4 semanas por mes) ===
    resultado = {}

    for variedad in df_pred["VARIEDAD"].unique():
        df_v = df_pred[df_pred["VARIEDAD"] == variedad].reset_index(drop=True)

        mes_1 = df_v.iloc[0:4].sum()
        mes_2 = df_v.iloc[4:8].sum()

        resultado[variedad] = [
            {"MES": "Mes 1", "HW": round(mes_1["HW"], 2), "LSTM": round(mes_1["LSTM"], 2)},
            {"MES": "Mes 2", "HW": round(mes_2["HW"], 2), "LSTM": round(mes_2["LSTM"], 2)},
        ]

    variedades = sorted(resultado.keys())
    variedad_sel = request.args.get("variedad")

    tabla_mensual = {}

    if variedad_sel:
        variedad_sel = variedad_sel.strip().upper()
        if variedad_sel in resultado:
            tabla_mensual = {variedad_sel: resultado[variedad_sel]}

    return render_template(
        "prediccion_tabla.html",
        tabla_semanal=tabla_mensual,
        variedades=variedades,
        variedad_sel=variedad_sel
    )

# ----------------------------
# PANTALLA ABAJO 2
# ----------------------------
@app.route('/prediccion_tabla_dos')
def prediccion_tabla_dos():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    # === VARIEDAD SELECCIONADA ===
    variedad_sel = request.args.get("variedad")

    df_original = pd.read_excel("produccion.xlsx")

    df_original["FECHA"] = pd.to_datetime(
        df_original["AÑO"].astype(str) + df_original["SEMANA"].astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )

    ultimo_año_real = df_original["AÑO"].max()
    ultima_semana_real = df_original[df_original["AÑO"] == ultimo_año_real]["SEMANA"].max()

    semanas_por_año = df_original.groupby("AÑO")["SEMANA"].nunique()
    años_completos = semanas_por_año[semanas_por_año >= 52].index
    df_historico = df_original[df_original["AÑO"].isin(años_completos)].copy()

    año_max_historico = df_historico["AÑO"].max()
    df_historico = df_historico[df_historico["AÑO"] >= año_max_historico - 5]

    filas = []

    for variedad in df_historico["VARIEDAD"].unique():

        if variedad_sel and variedad != variedad_sel:
            continue

        semana_actual_pred = ultima_semana_real + 1
        año_actual_pred = ultimo_año_real

        for _ in range(8):

            semana = semana_actual_pred
            año = año_actual_pred

            if semana > 52:
                semana -= 52
                año += 1

            df_sem = df_historico[
                (df_historico["VARIEDAD"] == variedad) &
                (df_historico["SEMANA"] == semana)
            ]

            serie = df_sem.sort_values("AÑO").set_index("AÑO")["TALLOS"].dropna()

            hw = round(serie.tail(2).mean(), 2) if len(serie) >= 2 else 0
            lstm = round(serie.tail(3).mean(), 2) if len(serie) >= 3 else 0

            filas.append([variedad, año, semana, hw, lstm])
            semana_actual_pred += 1

    df_pred = pd.DataFrame(filas, columns=["VARIEDAD", "AÑO", "SEMANA", "HW", "LSTM"])

    tabla_semanal = {}
    for v in df_pred["VARIEDAD"].unique():
        tabla_semanal[v] = df_pred[df_pred["VARIEDAD"] == v] \
            .drop(columns="VARIEDAD") \
            .to_dict(orient="records")

    tabla_semanal = dict(sorted(tabla_semanal.items()))

    variedades = sorted(df_historico["VARIEDAD"].unique())

    return render_template(
        "prediccion_tabla_dos.html",
        tabla_semanal=tabla_semanal if variedad_sel else {},
        variedades=variedades,
        variedad_sel=variedad_sel,
        max_semana=ultima_semana_real
    )


# ----------------------------
# PANTALLA 3 ARRIBA
# ----------------------------

@app.route('/prediccion_grafica', methods=['GET', 'POST'])
def prediccion_grafica():
    if 'usuario' not in session:
        return redirect(url_for('login'))

    import pandas as pd

    # =============================
    # 1. Selección de rango dinámico
    # =============================
    limite_semanas = 52
    rango_seleccionado = '1-52'

    if request.method == 'POST':
        rango_seleccionado = request.form.get('rango_semanas')
        try:
            limite_semanas = int(rango_seleccionado.split('-')[1])
        except:
            limite_semanas = 52

    # =============================
    # 2. Leer Excel
    # =============================
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

    # =============================
    # 3. Predicciones (SIN CAMBIOS)
    # =============================
    filas = []

    for variedad in df["VARIEDAD"].unique():
        for semana in range(1, limite_semanas + 1):

            df_sem = df[
                (df["VARIEDAD"] == variedad) &
                (df["SEMANA"] == semana)
            ]

            serie_full = (
                df_sem.sort_values("AÑO")
                .set_index("AÑO")["TALLOS"]
                .dropna()
            )

            serie_hw = serie_full[serie_full.index >= año_max - 1]
            serie_lstm = serie_full[serie_full.index >= año_max - 2]

            if len(serie_full) < 3:
                hw = 0
                lstm = 0
            else:
                hw = round(serie_hw.mean(), 2) if len(serie_hw) > 0 else 0
                lstm = round(serie_lstm.mean(), 2) if len(serie_lstm) > 0 else 0

            filas.append([variedad, semana, hw, lstm])

    df_pred = pd.DataFrame(filas, columns=["VARIEDAD", "SEMANA", "HW", "LSTM"])

    if df_pred.empty:
        return render_template(
            "prediccion_grafica.html",
            error="No se generaron datos.",
            rango_actual=rango_seleccionado
        )

    # =============================
    # 4. TABLAS (SIN CAMBIOS)
    # =============================

    tabla_semanal = {
        v: df_pred[df_pred["VARIEDAD"] == v]
            .drop(columns="VARIEDAD")
            .to_dict(orient="records")
        for v in df_pred["VARIEDAD"].unique()
    }

    df_pred["BLOQUE"] = ((df_pred["SEMANA"] - 1) // 4) + 1

    tabla_variedad_df = df_pred.groupby(["VARIEDAD", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()
    tabla_variedad = {
        v: tabla_variedad_df[tabla_variedad_df["VARIEDAD"] == v]
            .drop(columns="VARIEDAD")
            .to_dict(orient="records")
        for v in tabla_variedad_df["VARIEDAD"].unique()
    }

    tabla_total = df_pred.groupby("VARIEDAD")[["HW", "LSTM"]].sum().reset_index()

    df_excel = pd.read_excel("produccion.xlsx")[["VARIEDAD", "TIPO", "COLOR"]].drop_duplicates()
    df_merge = df_pred.merge(df_excel, on="VARIEDAD", how="left")

    tabla_tipo = df_merge.groupby("TIPO")[["HW", "LSTM"]].sum().reset_index()
    tabla_tipo = tabla_tipo[tabla_tipo["TIPO"].isin(["COLORES", "ROJO"])]

    tabla_general_semanal = df_pred.groupby("SEMANA")[["HW", "LSTM"]].sum().reset_index()
    tabla_general_bloques = df_pred.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    df_colores = df_merge[df_merge["TIPO"] == "COLORES"]
    tabla_general_colores = df_colores.groupby("BLOQUE")[["HW", "LSTM"]].sum().reset_index()

    tabla_general_total = pd.DataFrame([df_pred[["HW", "LSTM"]].sum().to_dict()])
    tabla_color_total = df_merge.groupby("COLOR")[["HW", "LSTM"]].sum().reset_index()
    tabla_color_bloques = df_merge.groupby(["COLOR", "BLOQUE"])[["HW", "LSTM"]].sum().reset_index()

    def get_color(value):
        return "#FFFFFF"

    max_semana = int(df_pred["SEMANA"].max())

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
# PANTALLA 3 ABAJO
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

