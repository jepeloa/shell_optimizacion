import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import seaborn as sns
import random

class TruckSchedulingModelORTools:
    def __init__(self):
        """
        Inicializa el modelo de programación de camiones utilizando OR-Tools
        """
        self.solver = None
        self.results = None
        self.schedule = None
        self.tiempos_con_vuelta = None
        self.tiempos_sin_vuelta = None
        self.tipos_camiones = None
        self.pedidos = None
        self.tiempos = {}
        self.debug = True
        # Coeficientes de penalización (ya no serán necesarios)
        # Las penalizaciones serán relativas y calculadas en el modelo
        self.total_penalizaciones = 0
        self.max_penalizaciones = {}

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
                        self.tiempos[cliente][fh] = {
                            'tiempo_entrega': 100.0,
                            'tiempo_total': 100.0
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
            'Tipo1': 20,  # 20 camiones tipo 1
            'Tipo2': 20,  # 20 camiones tipo 2
            'Tipo3': 20   # 20 camiones tipo 3
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
        
        # Generar exactamente 30 pedidos
        num_pedidos = 10
        for i in range(num_pedidos):
            cliente = random.choice(clientes)
            
            if self.debug:
                print(f"\nGenerando pedido {i+1} para cliente {cliente}")
                print(f"Franjas disponibles: {list(self.tiempos[cliente].keys())}")
            
            # Asignar tipo de camión de manera más uniforme
            tipo_camion = random.choice(tipos_disponibles)
            
            # Asignar franja horaria y grados de libertad
            franja_base = random.choice(['FH_1', 'FH_2', 'FH_3', 'FH_4'])
            grados_libertad = 3  # Máximo 3 grados de libertad
            
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
        Crea el modelo de programación lineal completo con todas las restricciones y penalizaciones relativas,
        incluyendo el cálculo del indicador de cumplimiento, utilizando OR-Tools.
        """
        if not self.verificar_datos():
            raise ValueError("Los datos no son válidos para crear el modelo")
        
        if self.debug:
            print("\nCreando modelo de programación lineal con penalizaciones relativas usando OR-Tools...")
            print(f"Número de pedidos a programar: {len(self.pedidos)}")
            print(f"Pedidos a programar: {self.pedidos['id_pedido'].tolist()}")
            print(f"Camiones disponibles: {[(t, n) for t, num in self.tipos_camiones.items() for n in range(num)]}")
        
        # Crear el solver
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self.solver:
            print('No se pudo crear el solver.')
            return
        
        # Conjuntos
        camiones = [(tipo, i) for tipo, num in self.tipos_camiones.items() 
                    for i in range(num)]
        pedidos = self.pedidos['id_pedido'].tolist()
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        
        # Mapeo de franjas a tiempos (en minutos)
        franja_tiempo = {
            'FH_1': (0, 360),       # 00:00 - 06:00
            'FH_2': (360, 720),     # 06:00 - 12:00
            'FH_3': (720, 1080),    # 12:00 - 18:00
            'FH_4': (1080, 1440)    # 18:00 - 24:00
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
        
        # Variables de holgura (ahora como atributos de la clase)
        self.slack_turno = {}       # Holgura para la restricción de turno
        self.slack_horas = {}       # Holgura para la restricción de horas máximas
        self.slack_franja = {}      # Holgura para la llegada fuera de franja
        self.slack_solapamiento = {}  # Holgura para el solapamiento de viajes
        
        # Variables para balanceo de pedidos por franja
        total_pedidos_franja = {}
        desviacion_franja = {}
        
        # Crear variables para camiones
        for t, n in camiones:
            u[t, n] = self.solver.BoolVar(f"u_{t}_{n}")  # Indica si el camión está en uso
            s[t, n] = self.solver.BoolVar(f"s_{t}_{n}")  # Indica el turno asignado al camión
        
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
                    y[p, t, n] = self.solver.BoolVar(f"y_{p}_{t}_{n}")
            
            # Variables para asignación de pedido a franja y tiempos
            for t, n in camiones:
                if t == tipo_requerido:
                    for f in franjas_validas:
                        x[p, t, n, f] = self.solver.BoolVar(f"x_{p}_{t}_{n}_{f}")
                        a_f, b_f = franja_tiempo[f]
                        tiempo_inicio[p, t, n, f] = self.solver.NumVar(
                            a_f, b_f, f"ti_{p}_{t}_{n}_{f}"
                        )
                        tiempo_fin[p, t, n, f] = self.solver.NumVar(
                            a_f, b_f + 300, f"tf_{p}_{t}_{n}_{f}"
                        )
                        # Variable auxiliar para la duración
                        duracion[p, t, n, f] = self.solver.NumVar(
                            0, self.solver.infinity(), f"dur_{p}_{t}_{n}_{f}"
                        )
                        # Variable de holgura para llegada fuera de franja
                        self.slack_franja[p, t, n, f] = self.solver.NumVar(
                            0, self.solver.infinity(), f"slack_franja_{p}_{t}_{n}_{f}"
                        )
        
        # Variables y restricciones para balanceo de pedidos por franja
        pedidos_promedio_por_franja = len(pedidos) / len(franjas)
        max_desviacion = len(pedidos) * (len(franjas) - 1) / len(franjas)
        self.max_penalizaciones['desbalance'] = max_desviacion
        for f in franjas:
            # Variable para total de pedidos en franja f
            total_pedidos_franja[f] = self.solver.NumVar(0, len(pedidos), f"total_pedidos_{f}")
            # Variable para desviación respecto al promedio
            desviacion_franja[f] = self.solver.NumVar(0, len(pedidos), f"desviacion_{f}")
            # Calcular total de pedidos en franja f
            self.solver.Add(total_pedidos_franja[f] == self.solver.Sum(
                x[p, t, n, f]
                for p in pedidos
                for t, n in camiones
                if (p, t, n, f) in x
            ))
            # Restricciones para calcular desviación absoluta
            self.solver.Add(total_pedidos_franja[f] - pedidos_promedio_por_franja <= desviacion_franja[f])
            self.solver.Add(pedidos_promedio_por_franja - total_pedidos_franja[f] <= desviacion_franja[f])
        
        # Restricciones (continúan como antes)
        # 1. Restricción de asignación única por pedido (mantener como restricción dura)
        for p in pedidos:
            tipo_requerido = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'tipo_camion'].iloc[0]
            # Cada pedido debe ser asignado exactamente a un camión
            self.solver.Add(
                sum(y[p, t, n] 
                    for t, n in camiones 
                    if t == tipo_requerido) == 1)
            
            # La suma de todas las asignaciones de franjas debe ser igual a 1
            self.solver.Add(
                sum(x[p, t, n, f]
                    for t, n in camiones
                    for f in franjas
                    if (p, t, n, f) in x) == 1)
            
            # Vincular variables y con x
            for t, n in camiones:
                if (p, t, n) in y:
                    self.solver.Add(
                        sum(x[p, t, n, f] for f in franjas if (p, t, n, f) in x) == y[p, t, n])

        # 2. Restricción de uso de camiones y asignación de turno (mantener como restricción dura)
        for t, n in camiones:
            # Si el camión está asignado a al menos un pedido, entonces está en uso
            self.solver.Add(u[t, n] >= sum(y[p, t, n] for p in pedidos if (p, t, n) in y))
            # Si el camión no está en uso, no se le asigna turno
            self.solver.Add(s[t, n] <= u[t, n])

        # 3. Restricción de turnos de trabajo (convertida en restricción suave)
        max_slack_turno = len(pedidos)
        self.max_penalizaciones['turno'] = max_slack_turno
        for t, n in camiones:
            for p in pedidos:
                if (p, t, n) in y:
                    for f in franjas:
                        if (p, t, n, f) in x:
                            # Variable de holgura para turno
                            self.slack_turno[p, t, n, f] = self.solver.NumVar(
                                0, 1, f"slack_turno_{p}_{t}_{n}_{f}"
                            )
                            # Restricción suavizada
                            if f in ['FH_1', 'FH_2']:
                                self.solver.Add(
                                    x[p, t, n, f] <= 1 - s[t, n] + self.slack_turno[p, t, n, f])
                            elif f in ['FH_3', 'FH_4']:
                                self.solver.Add(
                                    x[p, t, n, f] <= s[t, n] + self.slack_turno[p, t, n, f])

        # 4. Restricción de límite de horas de trabajo por camión (convertida en restricción suave)
        max_slack_horas = len(camiones) * 720  # Asumiendo que cada camión puede excederse en 12 horas
        self.max_penalizaciones['horas'] = max_slack_horas
        for t, n in camiones:
            total_tiempo_trabajo = sum(
                duracion[p, t, n, f]
                for p in pedidos for f in franjas if (p, t, n, f) in duracion
            )
            # Variable de holgura para horas extra
            self.slack_horas[t, n] = self.solver.NumVar(0, self.solver.infinity(), f"slack_horas_{t}_{n}")
            self.solver.Add(total_tiempo_trabajo <= 720 * u[t, n] + self.slack_horas[t, n])

        # 5. Restricción de llegada dentro de la franja horaria (convertida en restricción suave)
        max_slack_franja = len(pedidos) * max([b - a for a, b in franja_tiempo.values()])
        self.max_penalizaciones['franja'] = max_slack_franja
        for p, t, n, f in x:
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_entrega = self.tiempos[cliente][f]['tiempo_entrega']
            a_f, b_f = franja_tiempo[f]
            
            # Tiempo de llegada al cliente
            llegada_cliente = tiempo_inicio[p, t, n, f] + tiempo_entrega
            
            # Variables de holgura para llegadas tempranas y tardías
            earliness = self.solver.NumVar(0, self.solver.infinity(), f"earliness_{p}_{t}_{n}_{f}")
            tardiness = self.solver.NumVar(0, self.solver.infinity(), f"tardiness_{p}_{t}_{n}_{f}")
            
            self.solver.Add(llegada_cliente >= a_f * x[p, t, n, f] - earliness)
            self.solver.Add(llegada_cliente <= b_f * x[p, t, n, f] + tardiness)
            
            # Acumular la holgura en slack_franja
            self.solver.Add(self.slack_franja[p, t, n, f] == earliness + tardiness)

        # 6. Restricción de duración del viaje (mantener como restricción dura)
        for p, t, n, f in x:
            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
            tiempo_total = self.tiempos[cliente][f]['tiempo_total']
            
            self.solver.Add(tiempo_fin[p, t, n, f] == tiempo_inicio[p, t, n, f] + tiempo_total * x[p, t, n, f])
            
            # Definir la variable auxiliar duracion[p, t, n, f]
            M = 1e6  # Constante grande para Big-M
            self.solver.Add(duracion[p, t, n, f] >= tiempo_fin[p, t, n, f] - tiempo_inicio[p, t, n, f] - M * (1 - x[p, t, n, f]))
            self.solver.Add(duracion[p, t, n, f] <= tiempo_fin[p, t, n, f] - tiempo_inicio[p, t, n, f] + M * (1 - x[p, t, n, f]))
            self.solver.Add(duracion[p, t, n, f] >= 0)
            self.solver.Add(duracion[p, t, n, f] <= M * x[p, t, n, f])

        # 7. Restricción de no solapamiento (convertida en restricción suave)
        num_pedidos_camion = max([len([p for p in pedidos if (p, t, n) in y]) for t, n in camiones])
        max_slack_solapamiento = num_pedidos_camion * (num_pedidos_camion - 1) / 2 * max([b - a for a, b in franja_tiempo.values()])
        self.max_penalizaciones['solapamiento'] = max_slack_solapamiento
        M = 1e6  # Constante grande para Big-M
        for t, n in camiones:
            pedidos_camion = [p for p in pedidos if (p, t, n) in y]
            for i, p1 in enumerate(pedidos_camion):
                for p2 in pedidos_camion[i+1:]:
                    for f1 in franjas:
                        for f2 in franjas:
                            if (p1, t, n, f1) in x and (p2, t, n, f2) in x:
                                # Variable de holgura para solapamiento
                                self.slack_solapamiento[p1, p2, t, n, f1, f2] = self.solver.NumVar(
                                    0, self.solver.infinity(), f"slack_solapamiento_{p1}_{p2}_{t}_{n}_{f1}_{f2}"
                                )
                                # Modificar restricción de no solapamiento
                                z = self.solver.BoolVar(f"z_{p1}_{p2}_{t}_{n}_{f1}_{f2}")
                                
                                # Pedido p1 antes de p2
                                self.solver.Add(
                                    tiempo_inicio[p2, t, n, f2] >= tiempo_fin[p1, t, n, f1] + delta - 
                                    M * (1 - z + 2 - x[p1, t, n, f1] - x[p2, t, n, f2]) - self.slack_solapamiento[p1, p2, t, n, f1, f2]
                                )
                                
                                # Pedido p2 antes de p1
                                self.solver.Add(
                                    tiempo_inicio[p1, t, n, f1] >= tiempo_fin[p2, t, n, f2] + delta - 
                                    M * (z + 2 - x[p1, t, n, f1] - x[p2, t, n, f2]) - self.slack_solapamiento[p1, p2, t, n, f1, f2]
                                )

        # 8. Función objetivo: minimizar las penalizaciones relativas
        # Penalizaciones relativas entre 0 y 100
        objetivo = 0
        # Penalización por desbalance
        objetivo += 25 * self.solver.Sum(desviacion_franja.values()) / self.max_penalizaciones['desbalance']
        # Penalización por turno
        objetivo += 25 * self.solver.Sum(self.slack_turno.values()) / self.max_penalizaciones['turno']
        # Penalización por horas extra
        objetivo += 25 * self.solver.Sum(self.slack_horas.values()) / self.max_penalizaciones['horas']
        # Penalización por llegada fuera de franja
        objetivo += 15 * self.solver.Sum(self.slack_franja.values()) / self.max_penalizaciones['franja']
        # Penalización por solapamiento
        if self.max_penalizaciones['solapamiento'] > 0:
            objetivo += 10 * self.solver.Sum(self.slack_solapamiento.values()) / self.max_penalizaciones['solapamiento']
        else:
            objetivo += 0
        self.solver.Minimize(objetivo)
        
        if self.debug:
            print("\nModelo creado exitosamente con penalizaciones relativas usando OR-Tools")
            print(f"Número de variables: {self.solver.NumVariables()}")
            print(f"Número de restricciones: {self.solver.NumConstraints()}")

        # Guardar variables para usarlas en otros métodos
        self.x = x
        self.y = y
        self.u = u
        self.s = s
        self.tiempo_inicio = tiempo_inicio
        self.tiempo_fin = tiempo_fin
        self.duracion = duracion
        self.desviacion_franja = desviacion_franja  # Para análisis posterior

    def resolver(self):
        """
        Resuelve el modelo de programación lineal y calcula el indicador de cumplimiento.
        """
        print("\nResolviendo el modelo...")
        
        # Configurar y resolver
        status = self.solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            print("Se encontró una solución óptima")
            valor_objetivo = self.solver.Objective().Value()
            print(f"Valor de la función objetivo (penalización total relativa): {valor_objetivo:.2f}")
            
            # Calcular el total de penalizaciones relativas
            total_penalizacion_desbalance = 25 * sum(var.solution_value() for var in self.desviacion_franja.values()) / self.max_penalizaciones['desbalance']
            total_penalizacion_turno = 25 * sum(var.solution_value() for var in self.slack_turno.values()) / self.max_penalizaciones['turno']
            total_penalizacion_horas = 25 * sum(var.solution_value() for var in self.slack_horas.values()) / self.max_penalizaciones['horas']
            total_penalizacion_franja = 15 * sum(var.solution_value() for var in self.slack_franja.values()) / self.max_penalizaciones['franja']
            if self.max_penalizaciones['solapamiento'] > 0:
                total_penalizacion_solapamiento = 10 * sum(var.solution_value() for var in self.slack_solapamiento.values()) / self.max_penalizaciones['solapamiento']
            else:
                total_penalizacion_solapamiento = 0

            total_penalizaciones = total_penalizacion_desbalance + total_penalizacion_turno + total_penalizacion_horas + total_penalizacion_franja + total_penalizacion_solapamiento

            indicador_cumplimiento = max(0, 100 - total_penalizaciones)
            print(f"Indicador de cumplimiento: {indicador_cumplimiento:.2f}/100")
            
            # Mostrar penalizaciones individuales
            print("\nPenalizaciones individuales:")
            print(f"- Penalización por desbalance: {total_penalizacion_desbalance:.2f}/25")
            print(f"- Penalización por turno: {total_penalizacion_turno:.2f}/25")
            print(f"- Penalización por horas extra: {total_penalizacion_horas:.2f}/25")
            print(f"- Penalización por llegada fuera de franja: {total_penalizacion_franja:.2f}/15")
            print(f"- Penalización por solapamiento: {total_penalizacion_solapamiento:.2f}/10")
            
            # Identificar restricciones no satisfechas
            restricciones_no_cumplidas = []
            for var in self.slack_turno.values():
                if var.solution_value() > 0:
                    restricciones_no_cumplidas.append(var.name())
            for var in self.slack_horas.values():
                if var.solution_value() > 0:
                    restricciones_no_cumplidas.append(var.name())
            for var in self.slack_franja.values():
                if var.solution_value() > 0:
                    restricciones_no_cumplidas.append(var.name())
            for var in self.slack_solapamiento.values():
                if var.solution_value() > 0:
                    restricciones_no_cumplidas.append(var.name())
            
            if restricciones_no_cumplidas:
                print("\nRestricciones no cumplidas:")
                for restriccion in restricciones_no_cumplidas:
                    print(f"- {restriccion}")
            else:
                print("Todas las restricciones se cumplieron correctamente")
        elif status == pywraplp.Solver.INFEASIBLE:
            print("El modelo es infactible - No se encontró solución válida")
        else:
            print(f"El solver terminó con estado: {status}")
        
        return status == pywraplp.Solver.OPTIMAL

    def procesar_resultados(self):
        """
        Procesa los resultados del modelo
        """
        if not self.solver or self.solver.Solve() != pywraplp.Solver.OPTIMAL:
            print("No hay resultados óptimos para procesar")
            return False
        
        print("\nProcesando resultados...")
        resultados = []
        
        # Procesar asignaciones
        for p in self.pedidos['id_pedido']:
            for t in self.tipos_camiones.keys():
                for n in range(self.tipos_camiones[t]):
                    for f in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
                        key = (p, t, n, f)
                        if key in self.x and self.x[key].solution_value() > 0.5:
                            cliente = self.pedidos.loc[self.pedidos['id_pedido'] == p, 'cliente'].iloc[0]
                            tiempo_inicio_val = self.tiempo_inicio[key].solution_value()
                            tiempo_fin_val = self.tiempo_fin[key].solution_value()
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
        for i in range(5):
            plt.axvline(x=i*6, color='gray', linestyle='--', alpha=0.5)
            if i < 4:
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
        model = TruckSchedulingModelORTools()
        
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
                
                # Mostrar desviaciones en la asignación por franja
                print("\nDesviaciones en la asignación de pedidos por franja:")
                for f in ['FH_1', 'FH_2', 'FH_3', 'FH_4']:
                    desviacion = model.desviacion_franja[f].solution_value()
                    print(f"{f}: Desviación = {desviacion}")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    # Rutas de los archivos Excel (ajustar según sea necesario)
    ruta_con_vuelta = './Tiempo_vuelta.xlsx'
    ruta_sin_vuelta = './Tiempo_ida.xlsx'
    
    ejecutar_ejemplo(ruta_con_vuelta, ruta_sin_vuelta)
