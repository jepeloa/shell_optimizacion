import networkx as nx
import streamlit as st
import pandas as pd
from io import BytesIO

from truck_scheduling_model import TruckSchedulingModel
from matrix_processor import MatrixProcessor

# Inicializar o recuperar los modelos desde el estado de sesión
if "truck_model" not in st.session_state:
    st.session_state.truck_model = TruckSchedulingModel()

if "matrix_processor" not in st.session_state:
    st.session_state.matrix_processor = MatrixProcessor()

truck_model = st.session_state.truck_model
matrix_processor = st.session_state.matrix_processor

# Título de la aplicación
st.title("Scheduling Camiones")

st.sidebar.image("./shell_logo.png", use_container_width=True)

# Sidebar para cargar archivos
st.sidebar.header("Cargar Archivos Excel")
ruta_con_vuelta = st.sidebar.file_uploader("Subir archivo de tiempos con vuelta", type=["xlsx"])
ruta_sin_vuelta = st.sidebar.file_uploader("Subir archivo de tiempos sin vuelta", type=["xlsx"])

# Botón en el sidebar para cargar tiempos
if st.sidebar.button("Cargar Tiempos"):
    if ruta_con_vuelta and ruta_sin_vuelta:
        try:
            truck_model.cargar_tiempos(
                ruta_con_vuelta=BytesIO(ruta_con_vuelta.read()),
                ruta_sin_vuelta=BytesIO(ruta_sin_vuelta.read())
            )
            st.session_state.truck_model.tiempos_cargados = True
            st.sidebar.success("Tiempos cargados correctamente")
        except Exception as e:
            st.sidebar.error(f"Error al cargar los tiempos: {str(e)}")
    else:
        st.sidebar.error("Por favor, suba ambos archivos Excel")

st.sidebar.header("Definición de 5 tipos de camiones")

# Para cada tipo de camión se definen:
# - Nombre del tipo
# - Cantidad de pedidos a generar
# - Cantidad de camiones disponibles
tipo1 = st.sidebar.text_input("Tipo de Camión 1", value="Tipo1")
cant1_ped = st.sidebar.number_input(f"Cantidad de pedidos para {tipo1}", min_value=1, value=5)
cant1_cam = st.sidebar.number_input(f"Cantidad de camiones disponibles para {tipo1}", min_value=1, value=5)

tipo2 = st.sidebar.text_input("Tipo de Camión 2", value="Tipo2")
cant2_ped = st.sidebar.number_input(f"Cantidad de pedidos para {tipo2}", min_value=1, value=5)
cant2_cam = st.sidebar.number_input(f"Cantidad de camiones disponibles para {tipo2}", min_value=1, value=5)

tipo3 = st.sidebar.text_input("Tipo de Camión 3", value="Tipo3")
cant3_ped = st.sidebar.number_input(f"Cantidad de pedidos para {tipo3}", min_value=1, value=5)
cant3_cam = st.sidebar.number_input(f"Cantidad de camiones disponibles para {tipo3}", min_value=1, value=5)

tipo4 = st.sidebar.text_input("Tipo de Camión 4", value="Tipo4")
cant4_ped = st.sidebar.number_input(f"Cantidad de pedidos para {tipo4}", min_value=1, value=5)
cant4_cam = st.sidebar.number_input(f"Cantidad de camiones disponibles para {tipo4}", min_value=1, value=5)

tipo5 = st.sidebar.text_input("Tipo de Camión 5", value="Tipo5")
cant5_ped = st.sidebar.number_input(f"Cantidad de pedidos para {tipo5}", min_value=1, value=5)
cant5_cam = st.sidebar.number_input(f"Cantidad de camiones disponibles para {tipo5}", min_value=1, value=5)

# Diccionario de pedidos por tipo
tipos_camiones_pedidos = {
    tipo1: cant1_ped,
    tipo2: cant2_ped,
    tipo3: cant3_ped,
    tipo4: cant4_ped,
    tipo5: cant5_ped
}

# Diccionario de camiones disponibles por tipo
tipos_camiones_disponibles = {
    tipo1: cant1_cam,
    tipo2: cant2_cam,
    tipo3: cant3_cam,
    tipo4: cant4_cam,
    tipo5: cant5_cam
}

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
        if not getattr(st.session_state.truck_model, "tiempos_cargados", False):
            st.error("Debe cargar los tiempos antes de generar pedidos")
        else:
            pedidos = truck_model.generar_datos_prueba(
                porcentaje_grados_libertad_random=porcentaje_grados_libertad_random,
                fixed_grados_libertad=fixed_grados_libertad,
                porcentaje_franja_random=porcentaje_franja_random,
                fixed_franja=fixed_franja,
                tipos_camiones=tipos_camiones_pedidos  # Aquí solo pasamos los pedidos
            )
            st.session_state.truck_model.pedidos_generados = True
            # Guardar la disponibilidad en el modelo para uso posterior
            st.session_state.truck_model.tipos_camiones_disponibles = tipos_camiones_disponibles

            st.success("Pedidos generados correctamente")
            st.write("### Pedidos Generados")
            st.dataframe(pedidos)
    except Exception as e:
        st.error(f"Error al generar pedidos: {str(e)}")

# Botón para calcular y resolver el modelo
if st.sidebar.button("Calcular"):
    try:
        if not getattr(st.session_state.truck_model, "tiempos_cargados", False) or not getattr(st.session_state.truck_model, "pedidos_generados", False):
            st.error("Debe cargar los tiempos y generar pedidos antes de calcular")
        else:
            pedidos = truck_model.pedidos
            tipos_camiones_ped = tipos_camiones_pedidos
            tipos_camiones_disp = st.session_state.truck_model.tipos_camiones_disponibles

            resultados_totales = []
            tiempos = truck_model.tiempos

            # Procesar por tipo de camión
            for t_camion, cant_ped in tipos_camiones_ped.items():
                subset_pedidos = pedidos[pedidos['tipo_camion'] == t_camion].copy()
                if subset_pedidos.empty:
                    continue

                st.write(f"### Procesando tipo de camión: {t_camion}")
                st.info(f"Resolviendo el modelo para {t_camion}...")

                temp_model = TruckSchedulingModel()
                temp_model.tiempos = tiempos
                temp_model.pedidos = subset_pedidos
                # Setear la disponibilidad de camiones para este tipo
                temp_model.tipos_camiones = {t_camion: tipos_camiones_disp[t_camion]}

                temp_model.crear_modelo()
                if temp_model.resolver():
                    if temp_model.procesar_resultados():
                        st.success(f"Solución encontrada para {t_camion}")

                        # Estadísticas para este tipo
                        schedule_tipo = temp_model.schedule
                        camiones_usados = schedule_tipo.groupby('Tipo_Camion')['Num_Camion'].nunique().iloc[0]
                        pedidos_asignados = schedule_tipo['Pedido'].nunique()
                        camiones_disponibles = tipos_camiones_disp[t_camion]

                        # Mostrar estadísticas por tipo antes del gráfico
                        st.write("### Estadísticas para", t_camion)
                        st.write(f"- Camiones disponibles: {camiones_disponibles}")
                        st.write(f"- Camiones usados: {camiones_usados}")
                        st.write(f"- Pedidos asignados: {pedidos_asignados}")

                        # Mostrar el gráfico para este tipo de camión
                        with st.spinner(f"Generando gráfico para {t_camion}..."):
                            figs = temp_model.visualizar_resultados()
                            for fig in figs:
                                st.plotly_chart(fig, use_container_width=True)

                        resultados_totales.append(schedule_tipo)
                    else:
                        st.warning(f"No se pudo procesar resultados para {t_camion}")
                else:
                    st.warning(f"No se encontró solución para el tipo de camión {t_camion}")

            if resultados_totales:
                schedule_final = pd.concat(resultados_totales, ignore_index=True)
                st.session_state.truck_model.schedule = schedule_final

                st.markdown("---")
                st.success("Cálculo completado para todos los tipos de camión. Resultados combinados:")

                # Mostrar tabla de asignaciones combinadas
                st.write("### Tabla de Asignaciones Combinadas")
                st.dataframe(schedule_final)

                # Estadísticas finales por tipo de camión
                st.write("### Estadísticas Finales por Tipo de Camión")
                final_stats = []
                for t_camion in tipos_camiones_ped:
                    # Evitar error si no hubo pedidos asignados a ese tipo
                    df_tipo = schedule_final[schedule_final['Tipo_Camion'] == t_camion]
                    if df_tipo.empty:
                        # Si no hubo asignaciones, 0 camiones usados, 0 pedidos asignados
                        camiones_usados = 0
                        pedidos_asignados = 0
                    else:
                        camiones_usados = df_tipo['Num_Camion'].nunique()
                        pedidos_asignados = df_tipo['Pedido'].nunique()

                    camiones_disponibles = tipos_camiones_disp[t_camion]
                    final_stats.append({
                        "Tipo_Camion": t_camion,
                        "Camiones_Disponibles": camiones_disponibles,
                        "Camiones_Usados": camiones_usados,
                        "Pedidos_Asignados": pedidos_asignados
                    })

                final_stats_df = pd.DataFrame(final_stats)
                st.dataframe(final_stats_df)

                # Mostrar visualización unificada
                st.write("### Visualización Unificada de Resultados")
                with st.spinner("Generando gráfico combinado..."):
                    figs_final = truck_model.visualizar_resultados(schedule_final)
                    for fig_final in figs_final:
                        st.plotly_chart(fig_final, use_container_width=True)

                # Control de disponibilidad final
                st.write("### Control de Disponibilidad Final")
                # Mostramos la comparación Camiones Disponibles vs Camiones Usados por tipo
                for idx, row in final_stats_df.iterrows():
                    t = row["Tipo_Camion"]
                    disp = row["Camiones_Disponibles"]
                    usados = row["Camiones_Usados"]
                    st.write(f"- {t}: {usados} camiones usados de {disp} disponibles.")

            else:
                st.error("No se encontraron soluciones para ninguno de los tipos de camiones especificados.")

    except Exception as e:
        st.error(f"Error durante el cálculo: {str(e)}")
