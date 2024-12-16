import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go

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

        # Lista para asegurar distribución exactamente igual al total de pedidos
        tipos_disponibles = []
        for tipo, cantidad in self.tipos_camiones.items():
            tipos_disponibles.extend([tipo] * cantidad)

        random.seed(42)

        # Calcular la cantidad de pedidos con grados de libertad random
        num_pedidos_random_gl = int(num_pedidos * porcentaje_grados_libertad_random / 100)
        indices_pedidos_gl = list(range(num_pedidos))
        random.shuffle(indices_pedidos_gl)
        indices_random_gl = indices_pedidos_gl[:num_pedidos_random_gl]

        # Calcular la cantidad de pedidos con franja random
        num_pedidos_random_fh = int(num_pedidos * porcentaje_franja_random / 100)
        indices_pedidos_fh = list(range(num_pedidos))
        random.shuffle(indices_pedidos_fh)
        indices_random_fh = indices_pedidos_fh[:num_pedidos_random_fh]

        for i in range(num_pedidos):
            cliente = random.choice(clientes)
            tipo_camion = random.choice(tipos_disponibles)
            tipos_disponibles.remove(tipo_camion)

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
        # Método vacio para compatibilidad
        pass

    def resolver(self):
        if not self.verificar_datos():
            print("Datos no válidos")
            return False

        max_passes = int(self.pedidos.shape[0])+10
        pedidos_originales = self.pedidos.copy()
        pedidos_restantes = pedidos_originales.copy()

        all_solutions = []

        for pass_num in range(1, max_passes+1):
            n_pedidos = len(pedidos_restantes)
            if n_pedidos == 0:
                print(f"No quedaron pedidos sin asignar luego de la pasada {pass_num - 1}.")
                break

            num_iteraciones = n_pedidos * 1000
            print(f"Iniciando pasada {pass_num} con {n_pedidos} pedidos y {num_iteraciones} iteraciones...")

            solution = self._resolver_montecarlo(pedidos_restantes.to_dict('records'), num_iteraciones=num_iteraciones)

            if solution is None or solution.empty:
                print(f"No se encontró ninguna asignación en la pasada {pass_num}.")
                break
            else:
                all_solutions.append(solution)
                assigned_pedidos = set(solution['Pedido'].unique())
                pedidos_restantes = pedidos_restantes[~pedidos_restantes['id_pedido'].isin(assigned_pedidos)]
                print(f"Pasada {pass_num} asignó {len(assigned_pedidos)} pedidos.")

                if len(pedidos_restantes) == 0:
                    print(f"Todos los pedidos fueron asignados al finalizar la pasada {pass_num}.")
                    break
                else:
                    print(f"Quedan {len(pedidos_restantes)} pedidos sin asignar después de la pasada {pass_num}.")

        if len(all_solutions) == 0:
            print("No se pudieron asignar pedidos en ninguna pasada.")
            return False
        else:
            final_schedule = pd.concat(all_solutions, ignore_index=True)
            final_schedule.drop_duplicates(subset=['Pedido'], inplace=True)

            self.schedule = final_schedule
            self._postprocesar_schedule()

            assigned = set(self.schedule['Pedido'].unique())
            all_pedidos = set(pedidos_originales['id_pedido'].unique())
            not_assigned = all_pedidos - assigned
            if len(not_assigned) > 0:
                print(f"Después de {max_passes} pasadas, todavía quedan {len(not_assigned)} pedidos sin asignar.")
            else:
                print(f"Se asignaron todos los pedidos en un máximo de {max_passes} pasadas.")

            return True

    def _resolver_montecarlo(self, pedidos, num_iteraciones=100000):
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        franja_tiempo = {
            'FH_1': (0, 360),
            'FH_2': (360, 720),
            'FH_3': (721, 1080),
            'FH_4': (1081, 1439)
        }

        camiones = [(tipo, i) for tipo, num in self.tipos_camiones.items() for i in range(num)]

        def evaluar_solucion(schedule):
            franja_counts = {f:0 for f in franjas}
            for s in schedule:
                franja_counts[s['Franja']] += 1
            max_orders = max(franja_counts.values()) if franja_counts else 0
            min_orders = min(franja_counts.values()) if franja_counts else 0
            return len(schedule), max_orders - min_orders

        def asignar_viaje(camion_schedule, f, tiempo_entrega, tiempo_total):
            a_f, b_f = franja_tiempo[f]
            camion_schedule.sort(key=lambda x: x[0])
            start_candidate = a_f
            while True:
                end_candidate = start_candidate + tiempo_total
                if end_candidate > b_f:
                    return None, None
                solapa = any(not (end_candidate <= st or start_candidate >= fn) for (st, fn) in camion_schedule)
                if not solapa:
                    return start_candidate, end_candidate
                else:
                    start_candidate = max(fn for st, fn in camion_schedule if not (end_candidate <= st or start_candidate >= fn))

        best_schedule = None
        best_num_pedidos = -1
        best_obj = float('inf')

        for _ in range(num_iteraciones):
            camiones_franjas = {}
            random.shuffle(pedidos)
            current_solution = []

            for pedido in pedidos:
                cliente = pedido['cliente']
                fh_principal = pedido['FH_principal']
                gl = pedido['grados_libertad']
                tipo_camion = pedido['tipo_camion']
                p_id = pedido['id_pedido']

                fh_num = int(fh_principal[-1])
                franjas_validas = [f'FH_{i}' for i in range(fh_num, min(5, fh_num + gl + 1))]
                f_chosen = random.choice(franjas_validas)

                candidatos_camiones = [(t, n) for (t,n) in camiones if t == tipo_camion]
                random.shuffle(candidatos_camiones)

                tiempo_entrega = self.tiempos[cliente][f_chosen]['tiempo_entrega']
                tiempo_total = self.tiempos[cliente][f_chosen]['tiempo_total']

                asignado = False
                for (t_c, n_c) in candidatos_camiones:
                    key = (t_c, n_c, f_chosen)
                    if key not in camiones_franjas:
                        camiones_franjas[key] = []
                    inicio, fin = asignar_viaje(camiones_franjas[key], f_chosen, tiempo_entrega, tiempo_total)
                    if inicio is not None:
                        camiones_franjas[key].append((inicio, fin))
                        current_solution.append({
                            'Pedido': p_id,
                            'Cliente': cliente,
                            'Tipo_Camion': t_c,
                            'Num_Camion': n_c,
                            'Franja': f_chosen,
                            'Tiempo_Inicio': inicio,
                            'Tiempo_Fin': fin,
                            'Tiempo_Entrega': tiempo_entrega,
                            'Tiempo_Total': tiempo_total
                        })
                        asignado = True
                        break

            num_pedidos_asignados, obj_val = evaluar_solucion(current_solution)

            if (num_pedidos_asignados > best_num_pedidos) or \
               (num_pedidos_asignados == best_num_pedidos and obj_val < best_obj):
                best_num_pedidos = num_pedidos_asignados
                best_obj = obj_val
                best_schedule = current_solution

        if best_schedule is None or len(best_schedule) == 0:
            return None

        schedule_df = pd.DataFrame(best_schedule)
        schedule_df.sort_values(['Franja', 'Tiempo_Inicio'], inplace=True)
        return schedule_df

    def _postprocesar_schedule(self):
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        nuevas_asignaciones = []
        for franja in franjas:
            asignaciones_franja = self.schedule[self.schedule['Franja'] == franja]
            if not asignaciones_franja.empty:
                asignaciones_franja = asignaciones_franja.reset_index(drop=True)
                asignaciones_franja['Num_Camion'] = asignaciones_franja.index
                nuevas_asignaciones.append(asignaciones_franja)

        if nuevas_asignaciones:
            self.schedule = pd.concat(nuevas_asignaciones, ignore_index=True)
            self.schedule.sort_values(['Franja', 'Tiempo_Inicio'], inplace=True)

        nuevas_asignaciones = []
        for franja in franjas:
            asignaciones_franja = self.schedule[self.schedule['Franja'] == franja]
            if asignaciones_franja.empty:
                continue
            asignaciones_franja = asignaciones_franja.sort_values('Tiempo_Inicio').reset_index(drop=True)
            camiones_franja = []
            for _, viaje in asignaciones_franja.iterrows():
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

    def procesar_resultados(self):
        if self.schedule is None or self.schedule.empty:
            print("No hay resultados para procesar")
            return False
        return True

    def visualizar_resultados(self, schedule_df=None):
        if schedule_df is None:
            schedule_df = self.schedule

        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        franja_tiempo = {
            'FH_1': (0, 360),
            'FH_2': (360, 720),
            'FH_3': (721, 1080),
            'FH_4': (1081, 1439)
        }

        # Gráfico inicial (distribución)
        fig_inicial = None
        if self.pedidos is not None and not self.pedidos.empty:
            grouped = self.pedidos.groupby(['tipo_camion', 'FH_principal']).size().reset_index(name='count')
            all_fhs = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
            tipos_existentes = grouped['tipo_camion'].unique()
            all_combinations = pd.MultiIndex.from_product([tipos_existentes, all_fhs], names=['tipo_camion', 'FH_principal'])
            grouped = grouped.set_index(['tipo_camion', 'FH_principal']).reindex(all_combinations, fill_value=0).reset_index()

            fig_inicial = px.bar(
                grouped, 
                x='FH_principal', 
                y='count', 
                color='tipo_camion', 
                title='Distribución Inicial de Pedidos (Sin Optimizar)',
                labels={'FH_principal':'Franja Horaria','count':'Cantidad de Pedidos','tipo_camion':'Tipo Camión'},
                barmode='group'
            )
        else:
            fig_inicial = go.Figure()
            fig_inicial.add_annotation(text="No hay datos de pedidos iniciales", showarrow=False)

        # Gráfico pseudo-Gantt inicial
        fig_inicial_gantt = go.Figure()
        if self.pedidos is not None and not self.pedidos.empty and self.tiempos:
            pedidos_unicos = self.pedidos['id_pedido'].unique()
            colores = px.colors.qualitative.Set3
            color_dict = dict(zip(pedidos_unicos, colores * ((len(pedidos_unicos) // len(colores)) + 1)))

            for _, row in self.pedidos.iterrows():
                p_id = row['id_pedido']
                cliente = row['cliente']
                f = row['FH_principal']
                tipo_camion = row['tipo_camion']

                a_f, b_f = franja_tiempo[f]
                tiempo_entrega = self.tiempos[cliente][f]['tiempo_entrega']
                tiempo_total = self.tiempos[cliente][f]['tiempo_total']

                tiempo_inicio_horas = a_f/60.0
                tiempo_entrega_horas = tiempo_entrega / 60.0
                tiempo_total_horas = tiempo_total / 60.0

                fig_inicial_gantt.add_trace(
                    go.Bar(
                        x=[tiempo_entrega_horas],
                        y=[p_id],
                        base=[tiempo_inicio_horas],
                        orientation='h',
                        marker=dict(color=color_dict[p_id]),
                        name=p_id,
                        hoverinfo='text',
                        text=(
                            f"Pedido: {p_id}<br>"
                            f"Cliente: {cliente}<br>"
                            f"FH Principal: {f}<br>"
                            f"Tipo Camion: {tipo_camion}<br>"
                            f"Inicio: {tiempo_inicio_horas:.2f}h<br>"
                            f"Fin: {(tiempo_inicio_horas+tiempo_entrega_horas):.2f}h"
                        )
                    )
                )

                tiempo_vuelta = tiempo_total_horas - tiempo_entrega_horas
                fig_inicial_gantt.add_trace(
                    go.Bar(
                        x=[tiempo_vuelta],
                        y=[p_id],
                        base=[tiempo_inicio_horas + tiempo_entrega_horas],
                        orientation='h',
                        marker=dict(color=color_dict[p_id], opacity=0.5),
                        showlegend=False,
                        hoverinfo='text',
                        text=f"Tiempo de vuelta: {tiempo_vuelta:.2f}h"
                    )
                )

            fig_inicial_gantt.update_layout(
                barmode='stack',
                title='Programación sin Optimizar (Pedidos en sus Franjas Originales)',
                xaxis=dict(title='Hora del día', range=[0, 24]),
                yaxis=dict(title='Pedidos'),
                legend_title_text='Pedidos'
            )
            for i in range(4):
                fig_inicial_gantt.add_shape(
                    type="line",
                    x0=i*6,
                    y0=-0.5,
                    x1=i*6,
                    y1=len(self.pedidos['id_pedido'].unique()) + 0.5,
                    line=dict(color="Gray", dash="dash")
                )
                fig_inicial_gantt.add_annotation(
                    x=i*6 + 3,
                    y=-1,
                    text=f'FH_{i+1}',
                    showarrow=False,
                    xanchor='center'
                )
        else:
            fig_inicial_gantt.add_annotation(text="No hay datos o tiempos para el Gantt inicial sin optimizar", showarrow=False)

        # Gráfico final optimizado
        if schedule_df is None or schedule_df.empty:
            print("No hay resultados para visualizar")
            fig_gantt = go.Figure()
            fig_gantt.add_annotation(text="No hay datos de schedule final", showarrow=False)
            return [fig_inicial, fig_inicial_gantt, fig_gantt]

        schedule_viz = schedule_df.copy()
        for col in ['Tiempo_Inicio', 'Tiempo_Fin', 'Tiempo_Entrega', 'Tiempo_Total']:
            schedule_viz[f'{col}_Horas'] = schedule_viz[col] / 60

        schedule_viz['Camion_ID'] = schedule_viz['Tipo_Camion'].astype(str) + "_" + schedule_viz['Num_Camion'].astype(str)
        schedule_viz['Tiempo_Inicio_Datetime'] = pd.to_datetime(schedule_viz['Tiempo_Inicio_Horas'], unit='h', origin=pd.Timestamp('today'))
        schedule_viz['Tiempo_Fin_Datetime'] = pd.to_datetime(schedule_viz['Tiempo_Fin_Horas'], unit='h', origin=pd.Timestamp('today'))

        pedidos_unicos = schedule_viz['Pedido'].unique()
        colores = px.colors.qualitative.Set3
        color_dict = dict(zip(pedidos_unicos, colores * ((len(pedidos_unicos) // len(colores)) + 1)))

        fig_gantt = go.Figure()

        for _, row in schedule_viz.iterrows():
            fig_gantt.add_trace(
                go.Bar(
                    x=[row['Tiempo_Entrega_Horas']],
                    y=[row['Camion_ID']],
                    base=[row['Tiempo_Inicio_Horas']],
                    orientation='h',
                    marker=dict(color=color_dict[row['Pedido']]),
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
            tiempo_vuelta = row['Tiempo_Total_Horas'] - row['Tiempo_Entrega_Horas']
            fig_gantt.add_trace(
                go.Bar(
                    x=[tiempo_vuelta],
                    y=[row['Camion_ID']],
                    base=[row['Tiempo_Inicio_Horas'] + row['Tiempo_Entrega_Horas']],
                    orientation='h',
                    marker=dict(color=color_dict[row['Pedido']], opacity=0.5),
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Tiempo de vuelta: {tiempo_vuelta:.2f}h"
                )
            )

        fig_gantt.update_layout(
            barmode='stack',
            title='Programación de Camiones (Optimizado)',
            xaxis=dict(title='Hora del día', range=[0, 24]),
            yaxis=dict(title='Camión (Tipo_Num)'),
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

        return [fig_inicial_gantt, fig_gantt]
