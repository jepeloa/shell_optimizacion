# app.py
import networkx as nx
import streamlit as st
import pandas as pd
from io import BytesIO

# Importar las clases necesarias
from truck_scheduling_model import TruckSchedulingModel
from matrix_processor import MatrixProcessor


# Inicializar o recuperar los modelos desde el estado de sesión
if "truck_model" not in st.session_state:
    st.session_state.truck_model = TruckSchedulingModel()

if "matrix_processor" not in st.session_state:
    st.session_state.matrix_processor = MatrixProcessor()


truck_model = st.session_state.truck_model
matrix_processor=st.session_state.matrix_processor

# Título de la aplicación
st.title("Schedulling Camiones")

# Crear una pestaña para cada funcionalidad



st.header("Programación de Camiones")
st.sidebar.image("./shell_logo.png", use_container_width=True)
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
                    st.success("Postprocesando pedidos")
                       
                    # Mostrar tabla de asignaciones
                    st.write("### Tabla de Asignaciones")
                    st.dataframe(truck_model.schedule)
                    print(truck_model.schedule)
                    #matrix=matrix_processor.generar_matriz_ordenes(truck_model.schedule)
                    #st.write(matrix)
                    
                    
                    #elements=matrix_processor.get_elements(matrix)
                    #possible_pairs = matrix_processor.find_possible_pairs(elements, 4)
                    #G = matrix_processor.build_graph(possible_pairs)
                    #matching = matrix_processor.find_max_matching(G)
                    #groups = matrix_processor.get_groups_info(matching)
                    # print("--- Grupos Seleccionados ---")
                    # if groups:
                    #     for idx, group in enumerate(groups, start=1):
                    #         el1, el2 = group
                    #         print(f"Grupo {idx}: Elemento en Fila {el1[0]}, Columna {el1[1]} y Fila {el2[0]}, Columna {el2[1]}")
                    # else:
                    #     print("No se seleccionaron grupos.")
                    #     print("\n")
                    # matrix_processor.plot_matrix_with_plotly(matrix, groups=groups, title="Grupos Seleccionados en la Matriz Generada")
    
                    
                    #st.write(elements)
                    #matrix_processor.get_elements()
                    
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

