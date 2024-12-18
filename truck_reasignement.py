##Camino facil
#(cisternado 1, Pedido 1)------>(cisternado 2, Pedido 1)------>Callback(nuevo calculo)----->Optimizacion (nuevo calculo) 

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value


class TruckReasignement:
    # Datos de ejemplo
    O = ['o3']
    T = ['t10','t11','t5','t6']
    P = ['P1','P2','P3','P4']

    C = {
        't10': [5, 5, 5, 6, 6, 9],
        't11': [5, 5, 6, 6, 7, 7],
        't5':  [5, 5, 6, 6, 6, 9],
        't6': [8,6,6,5,5,6]
    }

    A = {
        't10': 0,
        't11': 0,
        't5' : 0,
        't6': 1
    }

    D = {
        # ('o1','P1'): 5,
        # ('o1','P2'): 5,
        # ('o1','P3'): 15,
        # ('o1','P4'): 11, # sum=36
        # ('o2','P1'): 10,
        # ('o2','P2'): 5,
        # ('o2','P3'): 9,
        # ('o2','P4'): 12  # sum=36
        ('o3','P1'):14,
        ('o3','P2'): 12,
        ('o3', 'P3'):10
    }

    # Crear el problema
    model = LpProblem("Reasignacion_con_Desviaciones", LpMinimize)

    # Variables
    x = {(o,t): LpVariable(f"x_{o}_{t}", cat=LpBinary) for o in O for t in T}
    z = {}
    for o in O:
        for t in T:
            for c_index, cap in enumerate(C[t]):
                for p in P:
                    z[(o,t,c_index,p)] = LpVariable(f"z_{o}_{t}_c{c_index}_{p}", cat=LpBinary)

    # Variables de desviación
    d_plus = {(o,t,p): LpVariable(f"dplus_{o}_{t}_{p}", lowBound=0) for o in O for t in T for p in P}
    d_minus = {(o,t,p): LpVariable(f"dminus_{o}_{t}_{p}", lowBound=0) for o in O for t in T for p in P}

    # Función objetivo: Minimizar la suma de desviaciones
    model += lpSum(d_plus[(o,t,p)] + d_minus[(o,t,p)] for o in O for t in T for p in P), "MinDesviaciones"

    # Restricciones

    # 1. Cada pedido asignado a un solo tipo
    for o in O:
        model += lpSum(x[(o,t)] for t in T) == 1, f"AsignaUnTipo_{o}"

    # 2. Disponibilidad
    for t in T:
        model += lpSum(x[(o,t)] for o in O) <= A[t], f"Disponibilidad_{t}"

    # 3. Cada cisterna se asigna a un producto si el camión se usa
    for o in O:
        for t in T:
            for c_index, cap in enumerate(C[t]):
                model += lpSum(z[(o,t,c_index,p)] for p in P) == x[(o,t)], f"CisternaUnProd_{o}_{t}_{c_index}"

    # 4. Desviación de demandas
    for o in O:
        for t in T:
            for p in P:
                # Cantidad asignada al producto p es demanda + desviaciones
                model += lpSum(C[t][c_index]*z[(o,t,c_index,p)] for c_index in range(len(C[t]))) \
                        == D.get((o,p),0)*x[(o,t)] + d_plus[(o,t,p)] - d_minus[(o,t,p)], f"Demanda_{o}_{t}_{p}"

    # (Opcional) Forzar desviaciones a cero si x=0 usando big-M
    M = 10
    for o in O:
        for t in T:
            for p in P:
                model += d_plus[(o,t,p)] <= M*x[(o,t)], f"LimiteDPlus_{o}_{t}_{p}"
                model += d_minus[(o,t,p)] <= M*x[(o,t)], f"LimiteDMinus_{o}_{t}_{p}"

    # Resolver
    model.solve()

    # Mostrar resultados
    print("Status:", model.status)

    for o in O:
        for t in T:
            if value(x[(o,t)]) > 0.5:
                print(f"Pedido {o} asignado al tipo {t}")
                # Mostrar desviaciones
                for p in P:
                    dp = value(d_plus[(o,t,p)])
                    dm = value(d_minus[(o,t,p)])
                    if dp > 1e-6 or dm > 1e-6:
                        print(f"  Producto {p}: Desviación +{dp}, -{dm}")
                # Mostrar asignación interna
                for c_index, cap in enumerate(C[t]):
                    for p in P:
                        if value(z[(o,t,c_index,p)]) > 0.5:
                            print(f"  Cisterna {c_index} (cap={cap}) -> {p}")