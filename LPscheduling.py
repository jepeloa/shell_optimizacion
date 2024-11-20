import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

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

    def cargar_tiempos(self, ruta_con_vuelta: str, ruta_sin_vuelta: str):
        """
        Carga los tiempos desde los archivos Excel
        """
        print("Cargando tiempos del LUNES...")
        
        # Cargar los dataframes
        self.tiempos_con_vuelta = pd.read_excel(ruta_con_vuelta)
        self.tiempos_sin_vuelta = pd.read_excel(ruta_sin_vuelta)
        
        # Limpiar nombres de columnas
        for df in [self.tiempos_con_vuelta, self.tiempos_sin_vuelta]:
            df.columns = [col.strip().upper() for col in df.columns]
        
        # Mapear las columnas del Excel específicamente para LUNES
        columnas_franjas = {
            'FH_1': 'LUNES FH_1',
            'FH_2': 'LUNES FH_2',
            'FH_3': 'LUNES FH_3',
            'FH_4': 'LUNES FH_4'
        }
        
        # Crear diccionario de tiempos
        self.tiempos = {}
        for _, row in self.tiempos_con_vuelta.iterrows():
            cliente = str(row['N° DE CLIENTE'])
            self.tiempos[cliente] = {}
            
            for fh in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
                col = columnas_franjas[fh]
                if col in row.index and not pd.isna(row[col]):
                    # Obtener tiempo con vuelta
                    tiempo_con = float(row[col])
                    
                    # Obtener tiempo sin vuelta
                    tiempo_sin = float(self.tiempos_sin_vuelta.loc[
                        self.tiempos_sin_vuelta['N° DE CLIENTE'] == int(cliente), col
                    ].iloc[0])
                    
                    if not pd.isna(tiempo_con) and not pd.isna(tiempo_sin):
                        self.tiempos[cliente][fh] = {
                            'tiempo_entrega': tiempo_sin,
                            'tiempo_total': tiempo_con
                        }
                else:
                    # Usar valor predeterminado de 100 minutos
                    self.tiempos[cliente][fh] = {
                        'tiempo_entrega': 100.0,
                        'tiempo_total': 100.0
                    }
        
        if self.debug:
            print(f"\nTiempos cargados para {len(self.tiempos)} clientes")
            print("\nEjemplo de tiempos para el primer cliente:")
            primer_cliente = list(self.tiempos.keys())[0]
            print(f"Cliente {primer_cliente}:")
            for fh, tiempos in self.tiempos[primer_cliente].items():
                print(f"  {fh}: {tiempos}")
        
        return self.tiempos
    
    def generar_datos_prueba(self):
        """
        Genera datos de prueba usando los tiempos reales cargados
        """
        if not self.tiempos:
            raise ValueError("Primero debe cargar los tiempos usando cargar_tiempos()")
        
        print("\nIniciando generación de datos de prueba...")
        
        # Definir tipos de camiones y su disponibilidad
        self.tipos_camiones = {
            'Tipo1': 20,  # 3 camiones tipo 1
            'Tipo2': 20,  # 2 camiones tipo 2
            'Tipo3': 20   # 2 camiones tipo 3
        }
        
        # Generar pedidos de prueba
        pedidos_data = []
        clientes = list(self.tiempos.keys())
        
        if self.debug:
            print(f"Clientes disponibles: {len(clientes)}")
        
        # Lista para asegurar distribución uniforme de tipos de camiones
        tipos_disponibles = []
        for tipo, cantidad in self.tipos_camiones.items():
            tipos_disponibles.extend([tipo] * cantidad)
        
        # Generar exactamente 10 pedidos
        num_pedidos = 30
        random.seed(42)
        for i in range(num_pedidos):
            cliente = random.choice(clientes)
            
            if self.debug:
                print(f"\nGenerando pedido {i+1} para cliente {cliente}")
                print(f"Franjas disponibles: {list(self.tiempos[cliente].keys())}")
            
            # Asignar tipo de camión de manera más uniforme
            tipo_camion = random.choice(tipos_disponibles)
            
            # Asignar franja horaria y grados de libertad
            franja_base = random.choice(['FH_1', 'FH_2', 'FH_3', 'FH_4'])
            grados_libertad = 3 #random.randint(0, 3)  # Máximo 2 grados de libertad
            
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
        
        return self.pedidos

    def verificar_datos(self):
        """
        Verifica la consistencia de los datos cargados
        """
        print("\nVerificando datos...")
        
        if self.pedidos is None or self.pedidos.empty:
            print("ERROR: No hay pedidos generados")
            return False
        
        if not self.tiempos:
            print("ERROR: No se han cargado los tiempos")
            return False
        
        print("\nVerificando consistencia de pedidos:")
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
        """
        Crea el modelo de programación lineal completo con todas las restricciones y correcciones aplicadas.
        """
        if not self.verificar_datos():
            raise ValueError("Los datos no son válidos para crear el modelo")
        
        if self.debug:
            print("\nCreando modelo de programación lineal con todas las correcciones...")
            print(f"Número de pedidos a programar: {len(self.pedidos)}")
            print(f"Pedidos a programar: {self.pedidos['id_pedido'].tolist()}")
            print(f"Camiones disponibles: {[(t, n) for t, num in self.tipos_camiones.items() for n in range(num)]}")
        
        self.model = LpProblem("Truck_Scheduling", LpMinimize)
        
        # Conjuntos
        camiones = [(tipo, i) for tipo, num in self.tipos_camiones.items() 
                    for i in range(num)]
        pedidos = self.pedidos['id_pedido'].tolist()
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        
        # Mapeo de franjas a tiempos (en minutos)
        franja_tiempo = {
            'FH_1': (0, 360),       # 00:00 - 06:00
            'FH_2': (360, 720),     # 06:00 - 12:00
            'FH_3': (721, 1080),    # 12:01 - 18:00
            'FH_4': (1081, 1439)    # 18:01 - 23:59
        }
        
        # Definir delta
        delta = 0  # Tiempo mínimo entre viajes (ajustar si es necesario)
        
        # Variables de decisión
        x = {}          # Variable binaria para asignación de pedido a camión y franja
        y = {}          # Variable binaria para asignación de pedido a camión
        u = {}          # Variable binaria para indicar si un camión está en uso
        s = {}          # Variable binaria para el turno asignado al camión
        tiempo_inicio = {}  # Tiempo de inicio del pedido
        tiempo_fin = {}     # Tiempo de fin del pedido
        duracion = {}       # Duración efectiva del viaje (variable auxiliar)
        
        # Crear variables para camiones
        for t, n in camiones:
            u[t, n] = LpVariable(f"u_{t}_{n}", 0, 1, LpBinary)  # Indica si el camión está en uso
            s[t, n] = LpVariable(f"s_{t}_{n}", 0, 1, LpBinary)  # Indica el turno asignado al camión
        
        # Crear variables para pedidos y asignaciones
        for p in pedidos:
            row = self.pedidos[self.pedidos['id_pedido'] == p].iloc[0]
            tipo_requerido = row['tipo_camion']
            fh_principal = row['FH_principal']
            gl = row['grados_libertad']
            
            # Determinar franjas válidas para el pedido considerando los grados de libertad
            fh_num = int(fh_principal[-1])
            franjas_validas = [f'FH_{i}' for i in range(
                max(1, fh_num - gl),
                min(5, fh_num + gl + 1)
            )]
            
            # Variables para asignación de pedido a camión
            for t, n in camiones:
                if t == tipo_requerido:
                    y[p, t, n] = LpVariable(f"y_{p}_{t}_{n}", 0, 1, LpBinary)
            
            # Variables para asignación de pedido a franja y tiempos
            for t, n in camiones:
                if t == tipo_requerido:
                    for f in franjas_validas:
                        x[p, t, n, f] = LpVariable(f"x_{p}_{t}_{n}_{f}", 0, 1, LpBinary)
                        a_f, b_f = franja_tiempo[f]
                        tiempo_inicio[p, t, n, f] = LpVariable(
                            f"ti_{p}_{t}_{n}_{f}",
                            lowBound=a_f,
                            upBound=b_f
                        )
                        tiempo_fin[p, t, n, f] = LpVariable(
                            f"tf_{p}_{t}_{n}_{f}",
                            lowBound=a_f,
                            upBound=b_f + 300  # Asumiendo que ningún viaje dura más de 300 minutos
                        )
                        # Variable auxiliar para la duración
                        duracion[p, t, n, f] = LpVariable(
                            f"dur_{p}_{t}_{n}_{f}",
                            lowBound=0
                        )
        
        # Restricciones
    
        # 1. Restricción de asignación única por pedido
        for p in pedidos:
            tipo_requerido = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'tipo_camion'].iloc[0]
            # Cada pedido debe ser asignado exactamente a un camión
            self.model += lpSum(y[p, t, n] 
                                for t, n in camiones 
                                if t == tipo_requerido) == 1, f"AssignTruck_{p}"
            
            # La suma de todas las asignaciones de franjas debe ser igual a 1
            self.model += lpSum(x[p, t, n, f]
                                for t, n in camiones
                                for f in franjas
                                if (p, t, n, f) in x) == 1, f"AssignSlot_{p}"
            
            # Vincular variables y con x
            for t, n in camiones:
                if (p, t, n) in y:
                    self.model += lpSum(x[p, t, n, f] for f in franjas if (p, t, n, f) in x) == y[p, t, n], \
                                f"Link_{p}_{t}_{n}"
        
        # 2. Restricción de uso de camiones y asignación de turno
        for t, n in camiones:
            # Si el camión está asignado a al menos un pedido, entonces está en uso
            self.model += u[t, n] >= lpSum(y[p, t, n] for p in pedidos if (p, t, n) in y), f"TruckUsage_{t}_{n}"
            # Si el camión no está en uso, no se le asigna turno
            self.model += s[t, n] <= u[t, n], f"ConsistenciaTurno_{t}_{n}"
        
        # 3. Restricción de turnos de trabajo
        for t, n in camiones:
            for p in pedidos:
                if (p, t, n) in y:
                    for f in franjas:
                        if (p, t, n, f) in x:
                            # Si f es FH1 o FH2, x <= 1 - s[t, n] (Turno 1)
                            if f in ['FH_1', 'FH_2']:
                                self.model += x[p, t, n, f] <= 1 - s[t, n], f"Turno1_{p}_{t}_{n}_{f}"
                            # Si f es FH3 o FH4, x <= s[t, n] (Turno 2)
                            elif f in ['FH_3', 'FH_4']:
                                self.model += x[p, t, n, f] <= s[t, n], f"Turno2_{p}_{t}_{n}_{f}"
        
        # 4. Restricción de límite de horas de trabajo por camión (máximo 720 minutos = 12 horas)
        for t, n in camiones:
            total_tiempo_trabajo = lpSum(
                duracion[p, t, n, f]
                for p in pedidos for f in franjas if (p, t, n, f) in duracion
            )
            self.model += total_tiempo_trabajo <= 720 * u[t, n], f"MaxHours_{t}_{n}"
        
        # 5. Restricción de llegada dentro de la franja horaria
        for p, t, n, f in x:
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_entrega = self.tiempos[cliente][f]['tiempo_entrega']
            a_f, b_f = franja_tiempo[f]
            
            # Tiempo de llegada al cliente
            llegada_cliente = tiempo_inicio[p, t, n, f] + tiempo_entrega
            
            self.model += llegada_cliente >= a_f * x[p, t, n, f], f"LlegadaMin_{p}_{t}_{n}_{f}"
            self.model += llegada_cliente <= b_f + (1 - x[p, t, n, f]) * 1e6, f"LlegadaMax_{p}_{t}_{n}_{f}"
        
        # 6. Restricción de duración del viaje (tiempo total con vuelta)
        for p, t, n, f in x:
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_total = self.tiempos[cliente][f]['tiempo_total']
            
            self.model += tiempo_fin[p, t, n, f] == tiempo_inicio[p, t, n, f] + tiempo_total * x[p, t, n, f], \
                        f"Duracion_{p}_{t}_{n}_{f}"
            
            # Definir la variable auxiliar duracion[p, t, n, f]
            M = 1e6  # Constante grande para Big-M
            self.model += duracion[p, t, n, f] >= tiempo_fin[p, t, n, f] - tiempo_inicio[p, t, n, f] - M * (1 - x[p, t, n, f]), \
                        f"DuracionMin_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] <= tiempo_fin[p, t, n, f] - tiempo_inicio[p, t, n, f] + M * (1 - x[p, t, n, f]), \
                        f"DuracionMax_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] >= 0, f"DuracionPos_{p}_{t}_{n}_{f}"
            self.model += duracion[p, t, n, f] <= M * x[p, t, n, f], f"DuracionZero_{p}_{t}_{n}_{f}"
        
        # 7. Restricción de no solapamiento considerando tiempo total con vuelta
        M = 1e6  # Constante grande para Big-M (asegúrate de que está definida)
        for t, n in camiones:
            pedidos_camion = [p for p in pedidos if (p, t, n) in y]
            for i, p1 in enumerate(pedidos_camion):
                for p2 in pedidos_camion[i+1:]:
                    for f1 in franjas:
                        for f2 in franjas:
                            if (p1, t, n, f1) in x and (p2, t, n, f2) in x:
                                z = LpVariable(f"z_{p1}_{p2}_{t}_{n}_{f1}_{f2}", 0, 1, LpBinary)
                                
                                # Pedido p1 antes de p2
                                self.model += tiempo_inicio[p2, t, n, f2] >= tiempo_fin[p1, t, n, f1] + delta - \
                                    M * (1 - z + 2 - x[p1, t, n, f1] - x[p2, t, n, f2]), f"NoOverlap1_{p1}_{p2}_{t}_{n}_{f1}_{f2}"
                                
                                # Pedido p2 antes de p1
                                self.model += tiempo_inicio[p1, t, n, f1] >= tiempo_fin[p2, t, n, f2] + delta - \
                                    M * (z + 2 - x[p1, t, n, f1] - x[p2, t, n, f2]), f"NoOverlap2_{p1}_{p2}_{t}_{n}_{f1}_{f2}"
        
        # 8. Función objetivo: minimizar la cantidad de camiones utilizados
        self.model += lpSum(u[t, n] for t, n in camiones), "MinimizeTrucksUsed"
        
        if self.debug:
            print("\nModelo creado exitosamente con todas las correcciones")
            print(f"Número de variables: {len(self.model.variables())}")
            print(f"Número de restricciones: {len(self.model.constraints)}")




    def resolver(self):
        """
        Resuelve el modelo de programación lineal
        """
        print("\nResolviendo el modelo...")
        
        # Configurar y resolver
        self.solver = self.model.solve()
        status = LpStatus[self.model.status]
        
        print(f"Estado del solver: {status}")
        
        if status == "Optimal":
            print("Se encontró una solución óptima")
            valor_objetivo = value(self.model.objective)
            print(f"Valor de la función objetivo: {valor_objetivo:.2f}")
        elif status == "Infeasible":
            print("El modelo es infactible - No se encontró solución válida")
        else:
            print(f"El solver terminó con estado: {status}")
        
        return status == "Optimal"

    def procesar_resultados(self):
        """
        Procesa los resultados del modelo
        """
        if not self.model or LpStatus[self.model.status] != "Optimal":
            print("No hay resultados óptimos para procesar")
            return False
        
        print("\nProcesando resultados...")
        resultados = []
        
        # Procesar asignaciones
        for var in self.model.variables():
            if var.value() > 0 and var.name.startswith("x_"):
                try:
                    # Extraer componentes del nombre
                    partes = var.name.split('_')
                    if len(partes) >= 5:
                        p = partes[1]  # id_pedido
                        t = partes[2]  # tipo_camion
                        n = partes[3]  # numero_camion
                        f = '_'.join(partes[4:])  # franja_horaria
                        
                        # Buscar cliente
                        cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
                        
                        # Buscar tiempos
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
                                'Num_Camion': n,
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
        
        # Crear DataFrame con resultados
        self.schedule = pd.DataFrame(resultados)
        self.schedule.sort_values(['Tipo_Camion', 'Num_Camion', 'Tiempo_Inicio'], 
                                inplace=True)
        
        print("\nResumen de asignaciones:")
        print(self.schedule.to_string())
        return True
    def visualizar_resultados(self):
        """
        Genera visualizaciones de los resultados
        """
        if self.schedule is None or self.schedule.empty:
            print("No hay resultados para visualizar")
            return
        
        print("\nGenerando visualizaciones...")
        
        # Convertir tiempos a horas para mejor visualización
        schedule_viz = self.schedule.copy()
        for col in ['Tiempo_Inicio', 'Tiempo_Fin', 'Tiempo_Entrega', 'Tiempo_Total']:
            schedule_viz[f'{col}_Horas'] = schedule_viz[col] / 60
        
        # 1. Diagrama de Gantt
        plt.figure(figsize=(15, 8))
        plt.title('Programación de Camiones')
        
        # Crear identificador único para cada camión
        camiones_unicos = schedule_viz.apply(
            lambda x: f"{x['Tipo_Camion']}_{x['Num_Camion']}", 
            axis=1
        ).unique()
        
        # Colores únicos para cada pedido
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.pedidos)))
        color_dict = dict(zip(self.pedidos['id_pedido'], colors))
        
        for i, camion in enumerate(camiones_unicos):
            datos_camion = schedule_viz[
                schedule_viz.apply(
                    lambda x: f"{x['Tipo_Camion']}_{x['Num_Camion']}", 
                    axis=1
                ) == camion
            ]
            
            for _, viaje in datos_camion.iterrows():
                # Barra principal para tiempo de entrega
                plt.barh(
                    i,
                    viaje['Tiempo_Entrega_Horas'],
                    left=viaje['Tiempo_Inicio_Horas'],
                    height=0.3,
                    color=color_dict[viaje['Pedido']],
                    label=f"{viaje['Pedido']} ({viaje['Cliente']})"
                )
                
                # Barra para tiempo de vuelta
                plt.barh(
                    i,
                    viaje['Tiempo_Total_Horas'] - viaje['Tiempo_Entrega_Horas'],
                    left=viaje['Tiempo_Inicio_Horas'] + viaje['Tiempo_Entrega_Horas'],
                    height=0.3,
                    color=color_dict[viaje['Pedido']],
                    alpha=0.3
                )
                
                # Etiqueta con información del viaje
                plt.text(
                    viaje['Tiempo_Inicio_Horas'],
                    i-0.15,
                    f"{viaje['Pedido']}\n{viaje['Cliente']}\n{viaje['Tiempo_Total_Horas']:.1f}h",
                    va='center',
                    fontsize=8
                )
        
        plt.yticks(range(len(camiones_unicos)), camiones_unicos)
        plt.xlabel('Hora del día')
        plt.xlim(0, 24)
        plt.grid(True)
        
        # Líneas verticales para franjas horarias
        for i in range(4):
            plt.axvline(x=i*6, color='gray', linestyle='--', alpha=0.5)
            plt.text(i*6 + 3, -0.8, f'FH_{i+1}', ha='center')
        
        # Eliminar duplicados en la leyenda
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(), 
            by_label.keys(),
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        
        plt.tight_layout()
        plt.show()
        
        # 2. Gráfico de utilización por tipo de camión
        plt.figure(figsize=(10, 6))
        utilizacion = schedule_viz.groupby('Tipo_Camion')['Tiempo_Total_Horas'].sum()
        utilizacion.plot(kind='bar')
        plt.title('Utilización por Tipo de Camión')
        plt.ylabel('Horas totales')
        plt.axhline(y=12, color='r', linestyle='--', label='Límite (12h)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 3. Distribución por franja horaria
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        schedule_viz['Franja'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Cantidad de Viajes por Franja')
        ax1.set_ylabel('Número de viajes')
        ax1.grid(True)
        
        schedule_viz.groupby('Franja')['Tiempo_Total_Horas'].sum().plot(kind='bar', ax=ax2)
        ax2.set_title('Tiempo Total por Franja')
        ax2.set_ylabel('Horas totales')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def ejecutar_ejemplo(ruta_con_vuelta: str, ruta_sin_vuelta: str):
    """
    Función principal para ejecutar el modelo
    """
    try:
        print("Iniciando ejecución del modelo...")
        
        # Crear instancia del modelo
        model = TruckSchedulingModel()
        
        # Cargar datos
        model.cargar_tiempos(ruta_con_vuelta, ruta_sin_vuelta)
        
        # Generar datos de prueba
        model.generar_datos_prueba()
        
        # Verificar datos
        if not model.verificar_datos():
            raise ValueError("Verificación de datos fallida")
        
        # Crear y resolver modelo
        model.crear_modelo()
        if model.resolver():
            # Procesar y visualizar resultados
            if model.procesar_resultados():
                model.visualizar_resultados()
                
                # Generar reportes adicionales
                print("\nGenerando reportes...")
                
                print("\nUtilización total por tipo de camión (horas):")
                print(model.schedule.groupby('Tipo_Camion')['Tiempo_Total'].sum() / 60)
                
                print("\nEstadísticas por franja horaria:")
                stats = model.schedule.groupby('Franja').agg({
                    'Pedido': 'count',
                    'Tiempo_Total': ['sum', 'mean']
                })
                stats.columns = ['Cantidad Viajes', 'Tiempo Total (min)', 'Promedio por Viaje (min)']
                print(stats)
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    # Rutas de los archivos Excel (ajustar según sea necesario)
    ruta_con_vuelta = './Tiempo_vuelta.xlsx'
    ruta_sin_vuelta = './Tiempo_ida.xlsx'
    
    ejecutar_ejemplo(ruta_con_vuelta, ruta_sin_vuelta)