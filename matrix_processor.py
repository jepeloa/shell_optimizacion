import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

class MatrixProcessor:
    def __init__(self):
        pass

    def generar_matriz_ordenes(self, df):
        """
        Genera una matriz donde las filas son camiones y las columnas son franjas horarias.
        Las columnas estarán siempre completas con FH_1 a FH_4 (renombradas como 0, 1, 2, 3),
        y cualquier columna extra será eliminada.

        :param df: DataFrame con columnas 'Num_Camion', 'Franja', y 'Pedido'.
        :return: DataFrame en formato de matriz con 4 columnas (0, 1, 2, 3).
        """
        # Asegurar que las franjas horarias sean 'FH_1', 'FH_2', 'FH_3', 'FH_4'
        franjas = ['FH_1', 'FH_2', 'FH_3', 'FH_4']
        max_camiones = df['Num_Camion'].max() + 1

        # Crear una matriz vacía con 0
        matriz = [[0 for _ in franjas] for _ in range(max_camiones)]

        # Completar la matriz con los conteos
        for _, row in df.iterrows():
            if row['Franja'] in franjas:
                franja_idx = franjas.index(row['Franja'])
                matriz[row['Num_Camion']][franja_idx] += 1

        # Convertir la matriz en un DataFrame
        matriz_df = pd.DataFrame(matriz, columns=[0, 1, 2, 3])

        return matriz_df

    def generate_matrix_from_orders(self, orders):
        """
        Genera una matriz basada en una lista de pedidos por columna.
        Cada pedido representa la suma total de la columna.
        Asegura que la cantidad de 1s sea igual a la de 2s tanto como sea posible.
        
        Parameters:
        - orders: list of int, pedidos por cada columna.
        
        Returns:
        - matrix: numpy.ndarray, matriz generada con 1s, 2s y 0s.
        """
        columns = []
        for idx, order in enumerate(orders):
            col_elements = self.generate_column_elements_balanced(order)
            columns.append(col_elements)
        
        max_rows = max(len(col) for col in columns)
        
        for i in range(len(columns)):
            if len(columns[i]) < max_rows:
                columns[i].extend([0] * (max_rows - len(columns[i])))
        
        matrix = np.array(columns).T
        return matrix

    def generate_column_elements_balanced(self, order):
        """
        Genera una columna balanceada con 1s y 2s según el pedido.
        """
        col_elements = []
        while order >= 3:
            col_elements.extend([1, 2])
            order -= 3
        if order == 2:
            col_elements.append(2)
        elif order == 1:
            col_elements.append(1)
        return col_elements

    def get_elements(self, matrix):
        """
        Obtiene una lista de elementos válidos (valor 1 o 2) con sus posiciones.
        Cada elemento se identifica por su (fila, columna).
        """
        elements = []
        if matrix.size == 0:
            return elements
        rows, cols = matrix.shape
        for r in range(rows):
            for c in range(cols):
                if matrix.iloc[r, c] in [1, 2]:
                    elements.append({'id': f"{r}_{c}", 'row': r, 'col': c, 'value': matrix.iloc[r, c]})
        return elements

    def find_possible_pairs(self, elements, cols):
        possible_pairs = []
        col_dict = {c: [] for c in range(cols)}
        for el in elements:
            col_dict[el['col']].append(el)
        
        for c in range(cols - 1):
            for el1 in col_dict[c]:
                for el2 in col_dict[c + 1]:
                    if el1['value'] + el2['value'] == 3:
                        possible_pairs.append((el1['id'], el2['id']))
        return possible_pairs

    def build_graph(self, possible_pairs):
        G = nx.Graph()
        G.add_edges_from(possible_pairs)
        return G

    def find_max_matching(self, G):
        matching = nx.max_weight_matching(G, maxcardinality=True)
        return matching

    def update_matrix(self, matrix, matching):
        updated_matrix = matrix.copy()
        for pair in matching:
            for node in pair:
                r, c = map(int, node.split('_'))
                updated_matrix.iloc[r, c] = 0
        return updated_matrix

    def get_groups_info(self, matching):
        groups = []
        for pair in matching:
            group = []
            for node in pair:
                r, c = map(int, node.split('_'))
                group.append((r + 1, c + 1))
            groups.append(group)
        return groups

    def plot_matrix_with_plotly(self, matrix, groups=None, title="Matriz"):
        rows, cols = matrix.shape
        fig = go.Figure()

        color_map = {0: 'white', 1: 'lightblue', 2: 'lightgreen'}
        z = matrix.values.tolist()
        colors = [[color_map[val] for val in row] for row in z]

        fig.add_trace(go.Heatmap(
            z=matrix.values,
            x=[c + 1 for c in range(cols)],
            y=[r + 1 for r in range(rows)],
            colorscale=[[0, 'white'], [0.5, 'lightblue'], [1, 'lightgreen']],
            showscale=False,
            hovertemplate='Fila %{y}, Col %{x}<br>Valor: %{z}<extra></extra>'
        ))

        annotations = []
        for r in range(rows):
            for c in range(cols):
                val = matrix.iloc[r, c]
                if val != 0:
                    annotations.append(dict(
                        x=c + 1,
                        y=r + 1,
                        text=str(val),
                        showarrow=False,
                        font=dict(color='black', size=12)
                    ))
        fig.update_layout(annotations=annotations)

        if groups:
            for group in groups:
                (r1, c1), (r2, c2) = group
                fig.add_shape(
                    type="line",
                    x0=c1,
                    y0=r1,
                    x1=c2,
                    y1=r2,
                    line=dict(color="red", width=2)
                )

        fig.update_layout(
            title=title,
            xaxis=dict(title='Columnas', tickmode='linear'),
            yaxis=dict(title='Filas', autorange='reversed', tickmode='linear'),
            width=max(600, cols * 60),
            height=max(800, rows * 60)
        )
        fig.show()
