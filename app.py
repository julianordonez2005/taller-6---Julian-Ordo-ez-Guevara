import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clasificador K-NN", page_icon="[o]", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
DIR_MODELOS = BASE_DIR / "modelos"


@st.cache_resource
def cargar(nombre_modelo, nombre_scaler, nombre_meta):
    requeridos = [nombre_modelo, nombre_scaler, nombre_meta]
    faltantes = [f for f in requeridos if not (DIR_MODELOS / f).exists()]
    if faltantes:
        st.error("No se encontraron los archivos del modelo.")
        st.write("**Faltantes:**", ", ".join(faltantes))
        st.write("**Ruta esperada:**", f"`{DIR_MODELOS}`")
        if DIR_MODELOS.exists():
            st.write("**Contenido de modelos/:**", sorted(p.name for p in DIR_MODELOS.iterdir()))
        else:
            st.write("La carpeta `modelos/` no existe en el repositorio.")
        st.stop()
    modelo = joblib.load(DIR_MODELOS / nombre_modelo)
    escalador = joblib.load(DIR_MODELOS / nombre_scaler)
    with open(DIR_MODELOS / nombre_meta, encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, escalador, meta


st.title("Clasificador K-NN sobre agrupaciones de K-Means")
tab_iris, tab_envios = st.tabs(["Iris Plant (caso canonico)", "Envios de e-commerce"])


# ----------------------------------------------------------------------
# IRIS
# ----------------------------------------------------------------------
with tab_iris:
    modelo, escalador, meta = cargar("knn_iris.joblib", "scaler_iris.joblib", "metadata_iris.json")

    st.caption(
        f"K-Means (k={meta['n_clusters']}, silueta={meta['silueta']}) + "
        f"K-NN (k={meta['n_vecinos']}, exactitud={meta['exactitud_knn']:.1%}, "
        f"ARI vs. especies reales={meta['ari_vs_especies']})"
    )

    st.subheader("Medidas morfologicas de la flor (cm)")
    r = meta["rangos"]
    valores = {}
    col1, col2 = st.columns(2)
    for i, var in enumerate(meta["variables"]):
        destino = col1 if i % 2 == 0 else col2
        with destino:
            valores[var] = st.slider(
                var, float(r[var][0]), float(r[var][1]),
                float((r[var][0] + r[var][1]) / 2), 0.1
            )

    if st.button("Clasificar flor", type="primary", use_container_width=True):
        entrada = pd.DataFrame([valores])[meta["variables"]]
        # .values evita el aviso de sklearn: el escalador se ajusto sobre un array
        grupo = int(modelo.predict(escalador.transform(entrada.values))[0])
        perfil = meta["perfiles"][str(grupo)]
        especie = perfil["especie"]
        pureza = perfil["pureza"]

        st.divider()
        st.subheader(f"Iris {especie}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Especie predominante", especie)
        c2.metric("Pureza del grupo", f"{pureza:.1%}")
        c3.metric("Muestras en el grupo", perfil["n_muestras"])

        if pureza == 1.0:
            st.success(
                f"Grupo puro: las {perfil['n_muestras']} flores que lo componen son "
                f"{especie}, sin ninguna excepcion."
            )
        else:
            otras = {k: v for k, v in perfil["composicion"].items() if k != especie}
            detalle = ", ".join(f"{v} {k}" for k, v in otras.items())
            st.warning(
                f"Clasificacion probable, no definitiva: el grupo contiene "
                f"{perfil['composicion'][especie]} flores {especie} y tambien {detalle}. "
                f"Versicolor y virginica se solapan y K-Means no logra separarlas por completo."
            )

        with st.expander("Composicion real del grupo"):
            st.dataframe(
                pd.DataFrame.from_dict(perfil["composicion"], orient="index",
                                       columns=["Numero de flores"])
            )

        with st.expander("Perfil promedio del grupo (centroide)"):
            st.dataframe(pd.DataFrame([perfil["centroide"]]).T.rename(columns={0: "Valor medio (cm)"}))

        st.caption(
            "La especie mostrada es la predominante en la agrupacion que K-Means encontro, "
            "no una identificacion botanica. El modelo agrupo las flores sin conocer sus "
            f"especies (ARI = {meta['ari_vs_especies']} frente a las reales): coincide de forma "
            "exacta con setosa, pero la frontera entre versicolor y virginica es geometrica "
            "y no biologica."
        )


# ----------------------------------------------------------------------
# ENVIOS
# ----------------------------------------------------------------------
with tab_envios:
    modelo_e, escalador_e, meta_e = cargar("knn_model.joblib", "scaler.joblib", "metadata.json")

    NOMBRES = {
        0: "Promociones agresivas / paquete liviano",
        1: "Envio pesado estandar",
        2: "Producto de alto valor / alta interaccion",
        3: "Cliente recurrente",
    }
    ACCIONES = {
        0: ["Reservar capacidad extra de despacho antes de la campana",
            "Asignar el transportista mas rapido disponible",
            "Notificar al cliente una fecha de entrega con holgura"],
        1: ["Consolidar con otros envios pesados para optimizar costo",
            "Ruta terrestre estandar",
            "Seguimiento normal"],
        2: ["Reforzar el embalaje por el alto valor del producto",
            "Activar seguimiento proactivo y avisos al cliente",
            "Priorizar la atencion en servicio al cliente"],
        3: ["Aplicar trato preferente por fidelidad",
            "Considerar envio sin costo o mejora de servicio",
            "Vigilar el cumplimiento para proteger la recompra"],
    }

    st.caption(
        f"K-Means (k={meta_e['n_clusters']}, silueta={meta_e['silueta']}) + "
        f"K-NN (k={meta_e['n_vecinos']}, exactitud={meta_e['exactitud_knn']:.1%})"
    )

    st.subheader("Datos del envio")
    re_ = meta_e["rangos"]
    col1, col2 = st.columns(2)
    with col1:
        peso = st.number_input("Peso del paquete (g)",
                               float(re_["Weight_in_gms"][0]), float(re_["Weight_in_gms"][1]), 3000.0, 50.0)
        costo = st.number_input("Costo del producto",
                                float(re_["Cost_of_the_Product"][0]), float(re_["Cost_of_the_Product"][1]), 210.0, 5.0)
        descuento = st.slider("Descuento ofrecido (%)",
                              int(re_["Discount_offered"][0]), int(re_["Discount_offered"][1]), 5)
    with col2:
        llamadas = st.slider("Llamadas a servicio al cliente",
                             int(re_["Customer_care_calls"][0]), int(re_["Customer_care_calls"][1]), 4)
        compras = st.slider("Compras previas del cliente",
                            int(re_["Prior_purchases"][0]), int(re_["Prior_purchases"][1]), 3)

    if st.button("Clasificar envio", type="primary", use_container_width=True):
        entrada = pd.DataFrame([{
            "Weight_in_gms": peso,
            "Cost_of_the_Product": costo,
            "Discount_offered": descuento,
            "Customer_care_calls": llamadas,
            "Prior_purchases": compras,
        }])[meta_e["variables"]]

        segmento = int(modelo_e.predict(escalador_e.transform(entrada))[0])
        perfil = meta_e["perfiles"][str(segmento)]
        riesgo = perfil["tasa_retraso"]

        st.divider()
        st.subheader(f"Segmento {segmento}: {NOMBRES.get(segmento, 'Sin nombre')}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Riesgo historico de retraso", f"{riesgo:.1%}",
                  delta=f"{(riesgo - meta_e['tasa_retraso_global']) * 100:+.1f} pp vs. global")
        c2.metric("Envios en el segmento", f"{perfil['n_envios']:,}")
        c3.metric("Peso del segmento", f"{perfil['porcentaje']}%")

        if riesgo >= 0.75:
            st.error("Riesgo ALTO. Este envio requiere intervencion preventiva.")
        elif riesgo >= 0.55:
            st.warning("Riesgo MEDIO. Conviene reforzar el seguimiento.")
        else:
            st.success("Riesgo MODERADO. Puede seguir el flujo estandar.")

        st.subheader("Acciones recomendadas")
        for accion in ACCIONES.get(segmento, []):
            st.write(f"- {accion}")

        with st.expander("Perfil promedio del segmento"):
            st.dataframe(pd.DataFrame([perfil["centroide"]]).T.rename(columns={0: "Valor medio"}))

        st.caption(
            "El riesgo mostrado es la tasa historica de retraso observada en el segmento, "
            "no una prediccion individual. Los segmentos se construyeron sin usar la "
            "variable de retraso."
        )
