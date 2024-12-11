import pandas as pd
import numpy as np
from pulp import *
from pulp import PULP_CBC_CMD
import plotly.express as px
import plotly.graph_objects as go
import random

class TruckSchedulingModel:
    def __init__(self):
        self.solver = None
        self.results = None
        self.schedule = None
        self.tiempos_con_vuelta = None
        self.tiempos_sin_vuelta = None
        self.tipos_camiones = None
        self.pedidos = None
        self.tiempos = {}
        self.model = None
        self.debug = True
        

    def cargar_tiempos(self, ruta_con_vuelta: str, ruta_sin_vuelta: str):
        print("Cargando tiempos del LUNES...")
        
        self.tiempos_con_vuelta = pd.read_excel(ruta_con_vuelta)
        self.tiempos_sin_vuelta = pd.read_excel(ruta_sin_vuelta)
        
        for df in [self.tiempos_con_vuelta, self.tiempos_sin_vuelta]:
            df.columns = [col.strip().upper() for col in df.columns]
        
        columnas_franjas = {
            'FH_1': 'LUNES FH_1',
            'FH_2': 'LUNES FH_2',
            'FH_3': 'LUNES FH_3',
            'FH_4': 'LUNES FH_4'
        }
        
        self.tiempos = {}
        for _, row in self.tiempos_con_vuelta.iterrows():
            cliente = str(row['N° DE CLIENTE'])
            self.tiempos[cliente] = {}
            
            for fh in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
                col = columnas_franjas[fh]
                if col in row.index and not pd.isna(row[col]):
                    tiempo_con = float(row[col])
                    
                    tiempo_sin = float(self.tiempos_sin_vuelta.loc[
                        self.tiempos_sin_vuelta['N° DE CLIENTE'] == int(cliente), col
                    ].iloc[0])
                    
                    if not pd.isna(tiempo_con) and not pd.isna(tiempo_sin):
                        self.tiempos[cliente][fh] = {
                            'tiempo_entrega': tiempo_sin,
                            'tiempo_total': tiempo_con
                        }
                else:
                    self.tiempos[cliente][fh] = {
                        'tiempo_entrega': 100.0,
                        'tiempo_total': 100.0
                    }
        
        return self.tiempos

    def generar_datos_prueba(self, porcentaje_grados_libertad_random=25, fixed_grados_libertad=1,
                             porcentaje_franja_random=25, fixed_franja='FH_1', 
                             tipos_camiones=None):
        if not self.tiempos:
            raise ValueError("Primero debe cargar los tiempos usando cargar_tiempos()")
        
        if tipos_camiones is None or len(tipos_camiones) == 0:
            raise ValueError("Debe especificar un diccionario con tipos de camiones y sus cantidades.")
        
        self.tipos_camiones = tipos_camiones
        num_pedidos = sum(tipos_camiones.values())

        # Generar pedidos de prueba
        pedidos_data = []
        clientes = list(self.tiempos.keys())

        # Lista para asegurar distribución uniforme de tipos de camiones
        tipos_disponibles = []
        for tipo, cantidad in self.tipos_camiones.items():
            tipos_disponibles.extend([tipo] * cantidad)

        random.seed(42)
        
        # Calcular la cantidad de pedidos con grados de libertad random y fijos
        num_pedidos_random_gl = int(num_pedidos * porcentaje_grados_libertad_random / 100)
        num_pedidos_fixed_gl = num_pedidos - num_pedidos_random_gl
        
        indices_pedidos_gl = list(range(num_pedidos))
        random.shuffle(indices_pedidos_gl)
        indices_random_gl = indices_pedidos_gl[:num_pedidos_random_gl]
        indices_fixed_gl = indices_pedidos_gl[num_pedidos_random_gl:]
        
        # Calcular la cantidad de pedidos con franja random y fijos
        num_pedidos_random_fh = int(num_pedidos * porcentaje_franja_random / 100)
        num_pedidos_fixed_fh = num_pedidos - num_pedidos_random_fh
        
        indices_pedidos_fh = list(range(num_pedidos))
        random.shuffle(indices_pedidos_fh)
        indices_random_fh = indices_pedidos_fh[:num_pedidos_random_fh]
        indices_fixed_fh = indices_pedidos_fh[num_pedidos_random_fh:]
        
        for i in range(num_pedidos):
            cliente = random.choice(clientes)
            
            # Asignar tipo de camión
            tipo_camion = random.choice(tipos_disponibles)
            # Mantenemos la estrategia para no sesgar demasiado
            tipos_disponibles.remove(tipo_camion)
            tipos_disponibles.append(tipo_camion)
            
            # Asignar grados de libertad
            if i in indices_random_gl:
                grados_libertad = random.randint(1, 3)
            else:
                grados_libertad = fixed_grados_libertad
            
            # Asignar franja horaria
            if i in indices_random_fh:
                franja_base = random.choice(['FH_1', 'FH_2', 'FH_3', 'FH_4'])
            else:
                franja_base = fixed_franja
            
            pedido = {
                'id_pedido': f'P{i+1}',
                'cliente': cliente,
                'tipo_camion': tipo_camion,
                'FH_principal': franja_base,
                'grados_libertad': grados_libertad
            }
            
            pedidos_data.append(pedido)
        
        self.pedidos = pd.DataFrame(pedidos_data)
        return self.pedidos

    def verificar_datos(self):
        if self.pedidos is None or self.pedidos.empty:
            return False
        if not self.tiempos:
            return False
        for _, pedido in self.pedidos.iterrows():
            cliente = pedido['cliente']
            fh = pedido['FH_principal']
            if cliente not in self.tiempos:
                return False
            if fh not in self.tiempos[cliente]:
                return False
        return True

    def crear_modelo(self):
        if not self.verificar_datos():
            raise ValueError("Los datos no son válidos para crear el modelo")

        self.model = LpProblem("Truck_Scheduling", LpMinimize)

        camiones = [(tipo, i) for tipo, num in self.tipos_camiones.items() for i in range(num)]
        pedidos = self.pedidos['id_pedido'].tolist()
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']

        franja_tiempo = {
            'FH_1': (0, 360),
            'FH_2': (360, 720),
            'FH_3': (721, 1080),
            'FH_4': (1081, 1439)
        }

        delta = 0
        x = {}
        y = {}
        tiempo_inicio = {}
        tiempo_fin = {}
        duracion = {}

        for p in pedidos:
            row = self.pedidos[self.pedidos['id_pedido'] == p].iloc[0]
            tipo_requerido = row['tipo_camion']
            fh_principal = row['FH_principal']
            gl = row['grados_libertad']

            fh_num = int(fh_principal[-1])
            franjas_validas = [f'FH_{i}' for i in range(fh_num, min(5, fh_num + gl + 1))]

            for t, n in camiones:
                if t == tipo_requerido:
                    y[p, t, n] = LpVariable(f"y_{p}_{t}_{n}", 0, 1, LpBinary)
                    for f in franjas_validas:
                        x[p, t, n, f] = LpVariable(f"x_{p}_{t}_{n}_{f}", 0, 1, LpBinary)
                        a_f, b_f = franja_tiempo[f]
                        tiempo_inicio[p, t, n, f] = LpVariable(f"ti_{p}_{t}_{n}_{f}", lowBound=a_f, upBound=b_f)
                        tiempo_fin[p, t, n, f] = LpVariable(f"tf_{p}_{t}_{n}_{f}", lowBound=a_f, upBound=b_f + 300)
                        duracion[p, t, n, f] = LpVariable(f"dur_{p}_{t}_{n}_{f}", lowBound=0)

        max_orders = LpVariable("max_orders", 0, None, LpInteger)
        min_orders = LpVariable("min_orders", 0, None, LpInteger)
        M = 2000

        # Restricciones
        for p in pedidos:
            tipo_requerido = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'tipo_camion'].iloc[0]
            self.model += lpSum(y[p, t, n] for t, n in camiones if t == tipo_requerido) == 1, f"AssignTruck_{p}"
            self.model += lpSum(x[p, t, n, f] for t, n in camiones for f in franjas if (p, t, n, f) in x) == 1, f"AssignSlot_{p}"

            for t, n in camiones:
                if (p, t, n) in y:
                    self.model += lpSum(x[p, t, n, f] for f in franjas if (p, t, n, f) in x) == y[p, t, n], f"Link_{p}_{t}_{n}"

        for f in franjas:
            self.model += lpSum(x[p, t, n, f] for p in pedidos for t, n in camiones if (p, t, n, f) in x) <= max_orders, f"MaxOrders_{f}"
            self.model += lpSum(x[p, t, n, f] for p in pedidos for t, n in camiones if (p, t, n, f) in x) >= min_orders, f"MinOrders_{f}"

        for p, t, n, f in x:
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_entrega = self.tiempos[cliente][f]['tiempo_entrega']
            a_f, b_f = franja_tiempo[f]
            llegada_cliente = tiempo_inicio[p, t, n, f] + tiempo_entrega
            self.model += llegada_cliente >= a_f * x[p, t, n, f], f"LlegadaMin_{p}_{t}_{n}_{f}"
            self.model += llegada_cliente <= b_f + (1 - x[p, t, n, f]) * M, f"LlegadaMax_{p}_{t}_{n}_{f}"

        for p, t, n, f in x:
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_total = self.tiempos[cliente][f]['tiempo_total']
            self.model += tiempo_fin[p, t, n, f] == tiempo_inicio[p, t, n, f] + tiempo_total * x[p, t, n, f], f"Duracion_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] >= tiempo_fin[p, t, n, f] - tiempo_inicio[p, t, n, f] - M * (1 - x[p, t, n, f]), f"DuracionMin_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] <= tiempo_fin[p, t, n, f] - tiempo_inicio[p, t, n, f] + M * (1 - x[p, t, n, f]), f"DuracionMax_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] >= 0, f"DuracionPos_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] <= M * x[p, t, n, f], f"DuracionZero_{p}_{t}_{n}_{f}"

        for t, n in camiones:
            pedidos_camion = [p for p in pedidos if (p, t, n) in y]
            for i, p1 in enumerate(pedidos_camion):
                for p2 in pedidos_camion[i+1:]:
                    for f1 in franjas:
                        for f2 in franjas:
                            if (p1, t, n, f1) in x and (p2, t, n, f2) in x:
                                z = LpVariable(f"z_{p1}_{p2}_{t}_{n}_{f1}_{f2}", 0, 1, LpBinary)
                                self.model += tiempo_inicio[p2, t, n, f2] >= tiempo_fin[p1, t, n, f1] + delta - M * (1 - z + 2 - x[p1, t, n, f1] - x[p2, t, n, f2])
                                self.model += tiempo_inicio[p1, t, n, f1] >= tiempo_fin[p2, t, n, f2] + delta - M * (z + 2 - x[p1, t, n, f1] - x[p2, t, n, f2])

        for p, t, n, f in x:
            if f in ['FH_1', 'FH_2']:
                self.model += tiempo_fin[p, t, n, f] <= 720 + M * (1 - x[p, t, n, f])
            elif f in ['FH_3', 'FH_4']:
                self.model += tiempo_fin[p, t, n, f] <= 1439 + M * (1 - x[p, t, n, f])

        self.model += max_orders - min_orders, "MinimizeSymmetryInFranja"

    def resolver(self):
        print("\nResolviendo el modelo...")
        cbc_solver = PULP_CBC_CMD(msg=True, timeLimit=3600, threads=8)
        self.solver = self.model.solve(cbc_solver)
        status = LpStatus[self.model.status]
        print(f"Estado del solver: {status}")

        return status == "Optimal"

    def procesar_resultados(self):
        if not self.model or LpStatus[self.model.status] != "Optimal":
            print("No hay resultados óptimos para procesar")
            return False
        
        resultados = []
        for var in self.model.variables():
            if var.value() > 0 and var.name.startswith("x_"):
                try:
                    partes = var.name.split('_')
                    if len(partes) >= 5:
                        p = partes[1]
                        t = partes[2]
                        n = partes[3]
                        f = '_'.join(partes[4:])
                        
                        cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
                        
                        tiempo_inicio_val = None
                        tiempo_fin_val = None
                        
                        for v in self.model.variables():
                            if v.name == f"ti_{p}_{t}_{n}_{f}" and v.value() is not None:
                                tiempo_inicio_val = v.value()
                            elif v.name == f"tf_{p}_{t}_{n}_{f}" and v.value() is not None:
                                tiempo_fin_val = v.value()
                        
                        if tiempo_inicio_val is not None and tiempo_fin_val is not None:
                            resultados.append({
                                'Pedido': p,
                                'Cliente': cliente,
                                'Tipo_Camion': t,
                                'Num_Camion': int(n),
                                'Franja': f,
                                'Tiempo_Inicio': tiempo_inicio_val,
                                'Tiempo_Fin': tiempo_fin_val,
                                'Tiempo_Entrega': self.tiempos[cliente][f]['tiempo_entrega'],
                                'Tiempo_Total': self.tiempos[cliente][f]['tiempo_total']
                            })
                except Exception as e:
                    print(f"Error procesando variable {var.name}: {str(e)}")
                    continue
        
        if not resultados:
            print("No se encontraron asignaciones válidas")
            return False
        
        self.schedule = pd.DataFrame(resultados)
        self.schedule.sort_values(['Franja', 'Tiempo_Inicio'], inplace=True)

        # Reasignar Num_Camion por franja para limpieza
        nuevas_asignaciones = []
        for franja in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
            asignaciones_franja = self.schedule[self.schedule['Franja'] == franja]
            if not asignaciones_franja.empty:
                asignaciones_franja = asignaciones_franja.reset_index(drop=True)
                asignaciones_franja['Num_Camion'] = asignaciones_franja.index
                nuevas_asignaciones.append(asignaciones_franja)

        if nuevas_asignaciones:
            self.schedule = pd.concat(nuevas_asignaciones, ignore_index=True)
            self.schedule.sort_values(['Franja', 'Tiempo_Inicio'], inplace=True)

        # Agrupar viajes contiguos
        nuevas_asignaciones = []
        for franja in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
            asignaciones_franja = self.schedule[self.schedule['Franja'] == franja]
            if asignaciones_franja.empty:
                continue
            asignaciones_franja = asignaciones_franja.sort_values('Tiempo_Inicio').reset_index(drop=True)
            camiones_franja = []
            for idx, viaje in asignaciones_franja.iterrows():
                asignado = False
                for camion in camiones_franja:
                    ultimo_viaje = camion['viajes'][-1]
                    if viaje['Tiempo_Inicio'] >= ultimo_viaje['Tiempo_Fin']:
                        camion['viajes'].append(viaje)
                        asignado = True
                        break
                if not asignado:
                    nuevo_camion = {'viajes': [viaje]}
                    camiones_franja.append(nuevo_camion)
            
            for num_camion, camion in enumerate(camiones_franja):
                for viaje in camion['viajes']:
                    viaje['Num_Camion'] = num_camion
                    nuevas_asignaciones.append(viaje)

        if nuevas_asignaciones:
            self.schedule = pd.DataFrame(nuevas_asignaciones)
            self.schedule.sort_values(['Franja', 'Num_Camion', 'Tiempo_Inicio'], inplace=True)

        return True

    def visualizar_resultados(self, schedule_df=None):
        """
        Genera visualizaciones de los resultados usando Plotly y devuelve las figuras para Streamlit.
        Ahora el Y-axis mostrará TipoCamion_NumCamion.
        Si se proporciona schedule_df, se usan esos datos en vez de self.schedule.
        """
        if schedule_df is None:
            schedule_df = self.schedule
        
        if schedule_df is None or schedule_df.empty:
            print("No hay resultados para visualizar")
            return []

        schedule_viz = schedule_df.copy()
        
        # Convertir tiempos a horas
        for col in ['Tiempo_Inicio', 'Tiempo_Fin', 'Tiempo_Entrega', 'Tiempo_Total']:
            schedule_viz[f'{col}_Horas'] = schedule_viz[col] / 60

        # Crear identificador Camion_ID como TipoCamion_NumCamion
        schedule_viz['Camion_ID'] = schedule_viz['Tipo_Camion'].astype(str) + "_" + schedule_viz['Num_Camion'].astype(str)

        # Convertir a datetime (para coherencia, aunque se use horas)
        schedule_viz['Tiempo_Inicio_Datetime'] = pd.to_datetime(schedule_viz['Tiempo_Inicio_Horas'], unit='h', origin=pd.Timestamp('today'))
        schedule_viz['Tiempo_Fin_Datetime'] = pd.to_datetime(schedule_viz['Tiempo_Fin_Horas'], unit='h', origin=pd.Timestamp('today'))

        # Colores para cada pedido
        pedidos_unicos = schedule_viz['Pedido'].unique()
        colores = px.colors.qualitative.Set3
        color_dict = dict(zip(pedidos_unicos, colores * (len(pedidos_unicos) // len(colores) + 1)))

        fig_gantt = go.Figure()

        for idx, row in schedule_viz.iterrows():
            fig_gantt.add_trace(
                go.Bar(
                    x=[row['Tiempo_Entrega_Horas']],
                    y=[row['Camion_ID']],
                    base=[row['Tiempo_Inicio_Horas']],
                    orientation='h',
                    marker=dict(
                        color=color_dict[row['Pedido']],
                    ),
                    name=row['Pedido'],
                    hoverinfo='text',
                    text=(
                        f"Pedido: {row['Pedido']}<br>"
                        f"Cliente: {row['Cliente']}<br>"
                        f"Inicio: {row['Tiempo_Inicio_Horas']:.2f}h<br>"
                        f"Fin: {row['Tiempo_Fin_Horas']:.2f}h<br>"
                        f"Tipo Camion: {row['Tipo_Camion']}"
                    )
                )
            )
            # Barra para el tiempo de vuelta
            tiempo_vuelta = row['Tiempo_Total_Horas'] - row['Tiempo_Entrega_Horas']
            fig_gantt.add_trace(
                go.Bar(
                    x=[tiempo_vuelta],
                    y=[row['Camion_ID']],
                    base=[row['Tiempo_Inicio_Horas'] + row['Tiempo_Entrega_Horas']],
                    orientation='h',
                    marker=dict(
                        color=color_dict[row['Pedido']],
                        opacity=0.5
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Tiempo de vuelta: {tiempo_vuelta:.2f}h"
                )
            )

        fig_gantt.update_layout(
            barmode='stack',
            title='Programación de Camiones',
            xaxis=dict(title='Hora del día', range=[0, 24]),
            yaxis=dict(title='Camión (Tipo_Num)'),
            legend_title_text='Pedidos'
        )

        # Añadir líneas verticales para franjas horarias
        for i in range(4):
            fig_gantt.add_shape(
                type="line",
                x0=i*6,
                y0=-0.5,
                x1=i*6,
                y1=len(schedule_viz['Camion_ID'].unique()) + 0.5,
                line=dict(color="Gray", dash="dash")
            )
            fig_gantt.add_annotation(
                x=i*6 + 3,
                y=-1,
                text=f'FH_{i+1}',
                showarrow=False,
                xanchor='center'
            )

        return [fig_gantt]
