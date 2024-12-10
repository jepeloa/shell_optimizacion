import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from ortools.linear_solver import pywraplp


class TruckSchedulingModel:
    def __init__(self):
        """
        Inicializa el modelo de programación de camiones
        """
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
        self.num_pedidos = None
        
        self.x = {}
        self.y = {}
        self.tiempo_inicio = {}
        self.tiempo_fin = {}
        self.duracion = {}
        self.max_orders_var = None
        self.min_orders_var = None
        self.status = None

    def cargar_tiempos(self, ruta_con_vuelta: str, ruta_sin_vuelta: str):
        """
        Carga los tiempos desde los archivos Excel
        """
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
        
        if self.debug:
            print(f"\nTiempos cargados para {len(self.tiempos)} clientes")
            primer_cliente = list(self.tiempos.keys())[0]
            print(f"Ejemplo de tiempos para cliente {primer_cliente}:")
            for fh, tiempos in self.tiempos[primer_cliente].items():
                print(f"  {fh}: {tiempos}")
        
        return self.tiempos

    def generar_datos_prueba(self, porcentaje_grados_libertad_random=25, fixed_grados_libertad=1,
                             porcentaje_franja_random=25, fixed_franja='FH_1', num_pedidos=20):
        if not self.tiempos:
            raise ValueError("Primero debe cargar los tiempos usando cargar_tiempos()")
        
        print("\nIniciando generación de datos de prueba...")
        
        self.tipos_camiones = {
            'Tipo1': num_pedidos,  
        }
        
        pedidos_data = []
        clientes = list(self.tiempos.keys())
        
        if self.debug:
            print(f"Clientes disponibles: {len(clientes)}")
        
        tipos_disponibles = []
        for tipo, cantidad in self.tipos_camiones.items():
            tipos_disponibles.extend([tipo] * cantidad)
        
        random.seed(42)
        
        num_pedidos_random_gl = int(num_pedidos * porcentaje_grados_libertad_random / 100)
        num_pedidos_fixed_gl = num_pedidos - num_pedidos_random_gl
        
        indices_pedidos_gl = list(range(num_pedidos))
        random.shuffle(indices_pedidos_gl)
        indices_random_gl = indices_pedidos_gl[:num_pedidos_random_gl]
        indices_fixed_gl = indices_pedidos_gl[num_pedidos_random_gl:]
        
        num_pedidos_random_fh = int(num_pedidos * porcentaje_franja_random / 100)
        num_pedidos_fixed_fh = num_pedidos - num_pedidos_random_fh
        
        indices_pedidos_fh = list(range(num_pedidos))
        random.shuffle(indices_pedidos_fh)
        indices_random_fh = indices_pedidos_fh[:num_pedidos_random_fh]
        indices_fixed_fh = indices_pedidos_fh[num_pedidos_random_fh:]
        
        for i in range(num_pedidos):
            cliente = random.choice(clientes)
            
            if self.debug:
                print(f"\nGenerando pedido {i+1} para cliente {cliente}")
            
            tipo_camion = random.choice(tipos_disponibles)
            tipos_disponibles.remove(tipo_camion)
            tipos_disponibles.append(tipo_camion)
            
            if i in indices_random_gl:
                grados_libertad = random.randint(1, 3)
            else:
                grados_libertad = fixed_grados_libertad
            
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
            
            if self.debug:
                print(f"Pedido generado: {pedido}")
            
            pedidos_data.append(pedido)
        
        self.pedidos = pd.DataFrame(pedidos_data)
        
        if self.debug:
            print("\nResumen de pedidos generados:")
            print(f"Total pedidos: {len(self.pedidos)}")
            print("\nDistribución por tipo de camión:")
            print(self.pedidos['tipo_camion'].value_counts())
            print("\nDistribución por franja horaria:")
            print(self.pedidos['FH_principal'].value_counts())
            print("\nDistribución por grados de libertad:")
            print(self.pedidos['grados_libertad'].value_counts())
        
        return self.pedidos

    def verificar_datos(self):
        print("\nVerificando datos...")
        
        if self.pedidos is None or self.pedidos.empty:
            print("ERROR: No hay pedidos generados")
            return False
        
        if not self.tiempos:
            print("ERROR: No se han cargado los tiempos")
            return False
        
        for _, pedido in self.pedidos.iterrows():
            cliente = pedido['cliente']
            fh = pedido['FH_principal']
            
            if cliente not in self.tiempos:
                print(f"ERROR: Cliente {cliente} no tiene tiempos definidos")
                return False
            
            if fh not in self.tiempos[cliente]:
                print(f"ERROR: No hay tiempos para cliente {cliente} en franja {fh}")
                return False
        
        print("Verificación completada exitosamente")
        return True

    def crear_modelo(self):
        if not self.verificar_datos():
            raise ValueError("Los datos no son válidos para crear el modelo")
        
        if self.debug:
            print("\nCreando modelo de programación entera mixta con OR-tools...")
            print(f"Número de pedidos a programar: {len(self.pedidos)}")
            print("Pedidos a programar:", self.pedidos['id_pedido'].tolist())
            print("Camiones disponibles:", [(t, n) for t, num in self.tipos_camiones.items() for n in range(num)])
        
        self.solver = pywraplp.Solver.CreateSolver('CBC')
        if not self.solver:
            raise ValueError("No se pudo crear el solver OR-tools")
        
        # Habilitar salida detallada del solver
        self.solver.EnableOutput()  
        # Podemos ajustar parámetros del solver si se desea más log interno:
        # self.solver.SetSolverSpecificParametersAsString('log=1')

        camiones = [(tipo, i) for tipo, num in self.tipos_camiones.items() 
                    for i in range(num)]
        pedidos = self.pedidos['id_pedido'].tolist()
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        
        franja_tiempo = {
            'FH_1': (0, 360),
            'FH_2': (360, 720),
            'FH_3': (721, 1080),
            'FH_4': (1081, 1439)
        }
        
        delta = 0
        M = 2000
        
        if self.debug:
            print("Creando variables...")
        
        for p in pedidos:
            row = self.pedidos[self.pedidos['id_pedido'] == p].iloc[0]
            tipo_requerido = row['tipo_camion']
            fh_principal = row['FH_principal']
            gl = row['grados_libertad']
            
            fh_num = int(fh_principal[-1])
            franjas_validas = [f'FH_{i}' for i in range(fh_num, min(5, fh_num + gl + 1))]
            
            for t, n in camiones:
                if t == tipo_requerido:
                    self.y[(p, t, n)] = self.solver.BoolVar(f"y_{p}_{t}_{n}")
                    for f in franjas_validas:
                        self.x[(p, t, n, f)] = self.solver.BoolVar(f"x_{p}_{t}_{n}_{f}")
                        a_f, b_f = franja_tiempo[f]
                        self.tiempo_inicio[(p, t, n, f)] = self.solver.NumVar(a_f, b_f, f"ti_{p}_{t}_{n}_{f}")
                        self.tiempo_fin[(p, t, n, f)] = self.solver.NumVar(a_f, b_f + 300, f"tf_{p}_{t}_{n}_{f}")
                        self.duracion[(p, t, n, f)] = self.solver.NumVar(0, M, f"dur_{p}_{t}_{n}_{f}")

        self.max_orders_var = self.solver.IntVar(0, self.solver.infinity(), "max_orders")
        self.min_orders_var = self.solver.IntVar(0, self.solver.infinity(), "min_orders")
        
        if self.debug:
            print("Creando restricciones...")

        # Restricciones
        for p in pedidos:
            self.solver.Add(
                sum(self.y[(p, t, n)] for t, n in camiones if (p, t, n) in self.y) == 1
            )
            self.solver.Add(
                sum(self.x[(p, t, n, f)] for t, n in camiones for f in franjas if (p, t, n, f) in self.x) == 1
            )
            for t, n in camiones:
                if (p, t, n) in self.y:
                    self.solver.Add(
                        sum(self.x[(p, t, n, f)] for f in franjas if (p, t, n, f) in self.x) == self.y[(p, t, n)]
                    )

        for f in franjas:
            self.solver.Add(
                sum(self.x[(p, t, n, f2)] for p in pedidos for t, n in camiones for f2 in [f] if (p, t, n, f2) in self.x)
                <= self.max_orders_var
            )
            self.solver.Add(
                sum(self.x[(p, t, n, f2)] for p in pedidos for t, n in camiones for f2 in [f] if (p, t, n, f2) in self.x)
                >= self.min_orders_var
            )

        for (p, t, n, f) in self.x.keys():
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_entrega = self.tiempos[cliente][f]['tiempo_entrega']
            a_f, b_f = franja_tiempo[f]
            llegada_cliente = self.solver.Sum([self.tiempo_inicio[(p, t, n, f)], tiempo_entrega])
            self.solver.Add(llegada_cliente >= a_f * self.x[(p, t, n, f)])
            self.solver.Add(llegada_cliente <= b_f + (1 - self.x[(p, t, n, f)]) * M)

        for (p, t, n, f) in self.x.keys():
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_total = self.tiempos[cliente][f]['tiempo_total']
            self.solver.Add(self.tiempo_fin[(p, t, n, f)] == self.solver.Sum([self.tiempo_inicio[(p, t, n, f)], tiempo_total * self.x[(p, t, n, f)]]))
            self.solver.Add(self.duracion[(p, t, n, f)] >= self.tiempo_fin[(p, t, n, f)] - self.tiempo_inicio[(p, t, n, f)] - M*(1 - self.x[(p, t, n, f)]))
            self.solver.Add(self.duracion[(p, t, n, f)] <= self.tiempo_fin[(p, t, n, f)] - self.tiempo_inicio[(p, t, n, f)] + M*(1 - self.x[(p, t, n, f)]))
            self.solver.Add(self.duracion[(p, t, n, f)] >= 0)
            self.solver.Add(self.duracion[(p, t, n, f)] <= M*self.x[(p, t, n, f)])

        for t, n in camiones:
            pedidos_camion = [p for p in pedidos if (p, t, n) in self.y]
            for i, p1 in enumerate(pedidos_camion):
                for p2 in pedidos_camion[i+1:]:
                    for f1 in franjas:
                        for f2 in franjas:
                            if (p1, t, n, f1) in self.x and (p2, t, n, f2) in self.x:
                                z = self.solver.BoolVar(f"z_{p1}_{p2}_{t}_{n}_{f1}_{f2}")
                                self.solver.Add(self.tiempo_inicio[(p2, t, n, f2)] >= 
                                                self.tiempo_fin[(p1, t, n, f1)] + delta - 
                                                M*(1 - z + 2 - self.x[(p1, t, n, f1)] - self.x[(p2, t, n, f2)]))
                                self.solver.Add(self.tiempo_inicio[(p1, t, n, f1)] >= 
                                                self.tiempo_fin[(p2, t, n, f2)] + delta - 
                                                M*(z + 2 - self.x[(p1, t, n, f1)] - self.x[(p2, t, n, f2)]))

        for (p, t, n, f) in self.x.keys():
            if f in ['FH_1', 'FH_2']:
                self.solver.Add(self.tiempo_fin[(p, t, n, f)] <= 720 + M*(1 - self.x[(p, t, n, f)]))
            else:
                self.solver.Add(self.tiempo_fin[(p, t, n, f)] <= 1439 + M*(1 - self.x[(p, t, n, f)]))

        self.solver.Minimize(self.solver.Sum([self.max_orders_var, -1*self.min_orders_var]))

        if self.debug:
            print("Modelo creado exitosamente.")
            print(f"Número de variables: {self.solver.NumVariables()}")
            print(f"Número de restricciones: {self.solver.NumConstraints()}")

    def resolver(self):
        print("\nResolviendo el modelo con OR-tools...")
        self.status = self.solver.Solve()
        
        if self.status == pywraplp.Solver.OPTIMAL:
            print("Se encontró una solución óptima")
            valor_objetivo = self.solver.Objective().Value()
            print(f"Valor de la función objetivo (max_orders - min_orders): {valor_objetivo}")
            return True
        elif self.status == pywraplp.Solver.INFEASIBLE:
            print("El modelo es infactible - No se encontró solución válida")
        else:
            print(f"El solver terminó con estado: {self.solver.StatusName(self.status)}")
        
        return False

    def procesar_resultados(self):
        if self.solver is None or self.status != pywraplp.Solver.OPTIMAL:
            print("No hay resultados óptimos para procesar")
            return False
        
        print("\nProcesando resultados...")
        resultados = []
        
        for (p, t, n, f), var in self.x.items():
            if var.solution_value() > 0.5:
                cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
                ti = self.tiempo_inicio[(p, t, n, f)].solution_value()
                tf = self.tiempo_fin[(p, t, n, f)].solution_value()
                tiempo_entrega = self.tiempos[cliente][f]['tiempo_entrega']
                tiempo_total = self.tiempos[cliente][f]['tiempo_total']
                
                resultados.append({
                    'Pedido': p,
                    'Cliente': cliente,
                    'Tipo_Camion': t,
                    'Num_Camion': int(n),
                    'Franja': f,
                    'Tiempo_Inicio': ti,
                    'Tiempo_Fin': tf,
                    'Tiempo_Entrega': tiempo_entrega,
                    'Tiempo_Total': tiempo_total
                })
        
        if not resultados:
            print("No se encontraron asignaciones válidas")
            return False
        
        self.schedule = pd.DataFrame(resultados)
        self.schedule.sort_values(['Franja', 'Tiempo_Inicio'], inplace=True)
        
        print("\nOptimización postprocesamiento: Eliminando filas en blanco y reestructurando asignaciones...")
        nuevas_asignaciones = []
        for franja in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
            asignaciones_franja = self.schedule[self.schedule['Franja'] == franja].copy()
            asignaciones_franja = asignaciones_franja.reset_index(drop=True)
            if not asignaciones_franja.empty:
                asignaciones_franja['Num_Camion'] = asignaciones_franja.index
                nuevas_asignaciones.append(asignaciones_franja)
        
        if nuevas_asignaciones:
            self.schedule = pd.concat(nuevas_asignaciones, ignore_index=True)
            self.schedule.sort_values(['Franja', 'Tiempo_Inicio'], inplace=True)

        print("\nAgrupando viajes contiguos en cada franja para optimizar el uso de camiones...")
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
        
        print("\nAsignaciones optimizadas:")
        print(self.schedule.to_string())
        return True

    def visualizar_resultados(self):
        if self.schedule is None or self.schedule.empty:
            print("No hay resultados para visualizar")
            return []
    
        print("\nGenerando visualizaciones con Plotly...")
    
        schedule_viz = self.schedule.copy()
        for col in ['Tiempo_Inicio', 'Tiempo_Fin', 'Tiempo_Entrega', 'Tiempo_Total']:
            schedule_viz[f'{col}_Horas'] = schedule_viz[col] / 60
    
        schedule_viz['Tiempo_Inicio_Datetime'] = pd.to_datetime(schedule_viz['Tiempo_Inicio_Horas'], unit='h', origin=pd.Timestamp('today'))
        schedule_viz['Tiempo_Fin_Datetime'] = pd.to_datetime(schedule_viz['Tiempo_Fin_Horas'], unit='h', origin=pd.Timestamp('today'))
        schedule_viz['Camion_ID'] = 'ID00' + schedule_viz['Num_Camion'].astype(str)
    
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
                    text=(f"Pedido: {row['Pedido']}<br>Cliente: {row['Cliente']}<br>"
                          f"Inicio: {row['Tiempo_Inicio_Horas']:.2f}h<br>Fin: {row['Tiempo_Fin_Horas']:.2f}h")
                )
            )
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
            yaxis=dict(title='Camión'),
            legend_title_text='Pedidos'
        )
    
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
    
        figures = [fig_gantt]
        return figures
