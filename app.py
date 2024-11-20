import streamlit as st
import pandas as pd
from io import BytesIO

# Importar la clase TruckSchedulingModel
from truck_scheduling_model import TruckSchedulingModel

# Inicializar o recuperar el modelo desde el estado de sesión
if "model" not in st.session_state:
    st.session_state.model = TruckSchedulingModel()

model = st.session_state.model

# Título de la aplicación
st.title("Programación de Camiones")

# Sidebar para cargar archivos
st.sidebar.header("Cargar Archivos Excel")
ruta_con_vuelta = st.sidebar.file_uploader("Subir archivo de tiempos con vuelta", type=["xlsx"])
ruta_sin_vuelta = st.sidebar.file_uploader("Subir archivo de tiempos sin vuelta", type=["xlsx"])

# Botón en el sidebar para cargar tiempos
if st.sidebar.button("Cargar Tiempos"):
    if ruta_con_vuelta and ruta_sin_vuelta:
        try:
            # Leer los archivos subidos
            model.cargar_tiempos(
                ruta_con_vuelta=BytesIO(ruta_con_vuelta.read()),
                ruta_sin_vuelta=BytesIO(ruta_sin_vuelta.read())
            )
            st.session_state.tiempos_cargados = True
            st.sidebar.success("Tiempos cargados correctamente")
        except Exception as e:
            st.sidebar.error(f"Error al cargar los tiempos: {str(e)}")
    else:
        st.sidebar.error("Por favor, suba ambos archivos Excel")

# Parámetros para generar pedidos
st.sidebar.header("Parámetros para Generar Pedidos")

porcentaje_grados_libertad_random = st.sidebar.number_input(
    "Porcentaje de grados de libertad aleatorios (%)", min_value=0, max_value=100, value=25
)
fixed_grados_libertad = st.sidebar.number_input(
    "Valor fijo para grados de libertad", min_value=1, max_value=3, value=1
)

porcentaje_franja_random = st.sidebar.number_input(
    "Porcentaje de franja horaria aleatoria (%)", min_value=0, max_value=100, value=25
)
fixed_franja = st.sidebar.selectbox(
    "Valor fijo para franja horaria", ['FH_1', 'FH_2', 'FH_3', 'FH_4'], index=0
)

# Botón para generar pedidos
if st.sidebar.button("Generar Pedidos"):
    try:
        if not getattr(st.session_state, "tiempos_cargados", False):
            st.error("Debe cargar los tiempos antes de generar pedidos")
        else:
            pedidos = model.generar_datos_prueba(
                porcentaje_grados_libertad_random=porcentaje_grados_libertad_random,
                fixed_grados_libertad=fixed_grados_libertad,
                porcentaje_franja_random=porcentaje_franja_random,
                fixed_franja=fixed_franja
            )
            st.session_state.pedidos_generados = True
            st.success("Pedidos generados correctamente")
            st.write("### Pedidos Generados")
            st.dataframe(pedidos)
    except Exception as e:
        st.error(f"Error al generar pedidos: {str(e)}")

# Botón para calcular y resolver el modelo
if st.sidebar.button("Calcular"):
    try:
        if not getattr(st.session_state, "tiempos_cargados", False) or not getattr(st.session_state, "pedidos_generados", False):
            st.error("Debe cargar los tiempos y generar pedidos antes de calcular")
        else:
            # Crear modelo y resolver
            model.crear_modelo()
            if model.resolver():
                if model.procesar_resultados():
                    st.success("Cálculo completado con éxito")

                    # Mostrar tabla de asignaciones
                    st.write("### Tabla de Asignaciones")
                    st.dataframe(model.schedule)

                    # Estadísticas generales
                    st.write("### Estadísticas Generales")
                    total_camiones = model.schedule['Num_Camion'].nunique()
                    total_pedidos = model.schedule['Pedido'].nunique()
                    st.write(f"- **Número total de camiones usados:** {total_camiones}")
                    st.write(f"- **Cantidad total de pedidos:** {total_pedidos}")

                    # Mostrar visualizaciones
                    st.write("### Visualización de Resultados")
                    with st.spinner("Generando gráficos..."):
                        figures = model.visualizar_resultados()  # Generar las figuras

                        # Mostrar cada figura en Streamlit
                        for fig in figures:
                            st.plotly_chart(fig, use_container_width=True)  # Renderizar la figura en Streamlit
    except Exception as e:
        st.error(f"Error durante el cálculo: {str(e)}")
