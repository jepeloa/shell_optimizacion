# app.py

import streamlit as st
import pandas as pd
from io import BytesIO

# Importar las clases necesarias
from truck_scheduling_model import TruckSchedulingModel
from matrix_balancer import MatrixBalancer

# Inicializar o recuperar los modelos desde el estado de sesión
if "truck_model" not in st.session_state:
    st.session_state.truck_model = TruckSchedulingModel()

if "matrix_balancer" not in st.session_state:
    st.session_state.matrix_balancer = None  # Inicialmente no hay balancer

truck_model = st.session_state.truck_model

# Título de la aplicación
st.title("Programación de Camiones y Balanceo de Matrices")

# Crear una pestaña para cada funcionalidad
tab1, tab2 = st.tabs(["Programación de Camiones", "Balanceo de Matrices"])

with tab1:
    st.header("Programación de Camiones")

    # Sidebar para cargar archivos
    st.sidebar.header("Cargar Archivos Excel")
    ruta_con_vuelta = st.sidebar.file_uploader("Subir archivo de tiempos con vuelta", type=["xlsx"])
    ruta_sin_vuelta = st.sidebar.file_uploader("Subir archivo de tiempos sin vuelta", type=["xlsx"])

    # Botón en el sidebar para cargar tiempos
    if st.sidebar.button("Cargar Tiempos"):
        if ruta_con_vuelta and ruta_sin_vuelta:
            try:
                # Leer los archivos subidos
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

    # Parámetros para generar pedidos
    st.sidebar.header("Parámetros para Generar Pedidos")

    # Menú desplegable de números del 5 al 50
    valor_seleccionado = st.sidebar.selectbox(
        "Seleccionar un número entero (5-50)",
        options=list(range(5, 51)),
        index=0
    )

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
                    num_pedidos=valor_seleccionado
                )
                st.session_state.truck_model.pedidos_generados = True
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
                # Crear modelo y resolver
                truck_model.crear_modelo()
                if truck_model.resolver():
                    if truck_model.procesar_resultados():
                        st.success("Cálculo completado con éxito")

                        # Mostrar tabla de asignaciones
                        st.write("### Tabla de Asignaciones")
                        st.dataframe(truck_model.schedule)

                        # Estadísticas generales
                        st.write("### Estadísticas Generales")
                        total_camiones = truck_model.schedule['Num_Camion'].nunique()
                        total_pedidos = truck_model.schedule['Pedido'].nunique()
                        st.write(f"- **Número total de camiones usados:** {total_camiones}")
                        st.write(f"- **Cantidad total de pedidos:** {total_pedidos}")

                        # Mostrar visualizaciones
                        st.write("### Visualización de Resultados")
                        with st.spinner("Generando gráficos..."):
                            figures = truck_model.visualizar_resultados()  # Generar las figuras

                            # Mostrar cada figura en Streamlit
                            for fig in figures:
                                st.plotly_chart(fig, use_container_width=True)  # Renderizar la figura en Streamlit
        except Exception as e:
            st.error(f"Error durante el cálculo: {str(e)}")

with tab2:
    st.header("Balanceo de Matrices Manual")

    st.write("""
    En esta sección, puedes ingresar manualmente la cantidad de 1s y 2s por cada columna para generar una matriz balanceada. 
    Luego, se encontrarán emparejamientos óptimos que suman 3 entre columnas contiguas.
    """)

    # Formulario para ingresar la cantidad de 1s y 2s por columna
    with st.form("balanceo_form"):
        st.subheader("Ingresar Cantidad de 1s y 2s por Columna")

        # Input para el número de columnas
        num_columnas = st.number_input("Número de Columnas", min_value=1, max_value=50, value=5, step=1)

        # Crear tablas dinámicas para ingresar 1s y 2s por columna
        ones_input = []
        twos_input = []
        for col in range(1, num_columnas + 1):
            cols = st.columns(2)
            with cols[0]:
                ones = st.number_input(f"Columna {col} - Número de 1s", min_value=0, value=1, step=1, key=f"ones_{col}")
            with cols[1]:
                twos = st.number_input(f"Columna {col} - Número de 2s", min_value=0, value=1, step=1, key=f"twos_{col}")
            ones_input.append(ones)
            twos_input.append(twos)

        submit_button = st.form_submit_button("Generar y Resolver Matriz")

    if submit_button:
        try:
            # Crear instancia de MatrixBalancer
            balancer = MatrixBalancer(ones_per_column=ones_input, twos_per_column=twos_input)
            st.session_state.matrix_balancer = balancer

            # Mostrar la matriz generada
            st.write("### Matriz Generada")
            df_matrix = balancer.get_matrix_as_dataframe()
            st.dataframe(df_matrix)

            # Visualizar la matriz
            fig_matrix = balancer.plot_matrix(title="Matriz Generada Basada en Entradas Manuales")
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Mostrar los grupos seleccionados
            st.write("---")
            st.write("### Grupos Seleccionados")
            if balancer.groups:
                for idx, group in enumerate(balancer.groups, start=1):
                    el1, el2 = group
                    st.write(f"**Grupo {idx}:** Elemento en Fila {el1[0]}, Columna {el1[1]} y Fila {el2[0]}, Columna {el2[1]}")
            else:
                st.write("No se seleccionaron grupos.")

            # Visualizar los grupos en la matriz
            if balancer.groups:
                fig_groups = balancer.plot_matrix(groups=balancer.groups, title="Grupos Seleccionados en la Matriz")
                st.plotly_chart(fig_groups, use_container_width=True)

            # Mostrar la matriz actualizada
            st.write("### Matriz Actualizada Después de Agrupamientos")
            df_updated = balancer.get_matrix_as_dataframe(updated=True)
            st.dataframe(df_updated)

            # Visualizar la matriz actualizada
            fig_updated = balancer.plot_matrix(updated=True, title="Matriz Actualizada (Grupos Eliminados)")
            st.plotly_chart(fig_updated, use_container_width=True)

            # Mostrar estadísticas
            total_elements, covered_elements = balancer.get_elements_count()
            total_groups = balancer.get_groups_count()
            st.write("---")
            st.write("### Estadísticas")
            st.write(f"- **Elementos Totales con Valor 1 o 2:** {total_elements}")
            st.write(f"- **Elementos Cubiertos por Grupos:** {covered_elements}")
            st.write(f"- **Cantidad de Grupos Utilizados:** {total_groups}")

        except Exception as e:
            st.error(f"Error al procesar la matriz: {str(e)}")
