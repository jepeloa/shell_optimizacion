# matrix_balancer.py

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import List, Tuple, Dict
import pandas as pd

class MatrixBalancer:
    def __init__(self, ones_per_column: List[int], twos_per_column: List[int]):
        """
        Inicializa la clase MatrixBalancer con la cantidad de 1s y 2s por columna.

        Parameters:
        - ones_per_column: List[int], número de 1s por columna.
        - twos_per_column: List[int], número de 2s por columna.
        """
        if len(ones_per_column) != len(twos_per_column):
            raise ValueError("Las listas de 1s y 2s por columna deben tener la misma longitud.")
        
        self.ones_per_column = ones_per_column
        self.twos_per_column = twos_per_column
        self.num_columns = len(ones_per_column)
        self.columns = self.generate_columns()
        self.matrix = self.generate_matrix()
        self.elements = self.get_elements()
        self.possible_pairs = self.find_possible_pairs()
        self.graph = self.build_graph()
        self.matching = self.find_max_matching()
        self.groups = self.get_groups_info()
        self.updated_matrix = self.update_matrix()

    def generate_columns(self) -> List[List[int]]:
        """
        Genera las columnas basadas en la cantidad de 1s y 2s por columna.

        Returns:
        - columns: List of lists, cada sublista representa una columna.
        """
        columns = []
        for idx, (num_ones, num_twos) in enumerate(zip(self.ones_per_column, self.twos_per_column)):
            col = [1] * num_ones + [2] * num_twos
            columns.append(col)
        return columns

    def generate_matrix(self) -> np.ndarray:
        """
        Genera una matriz numpy a partir de las columnas, rellenando con 0s donde sea necesario.

        Returns:
        - matrix: numpy.ndarray, matriz de filas x columnas.
        """
        max_rows = max(len(col) for col in self.columns) if self.columns else 0
        # Rellenar columnas con 0s
        for col in self.columns:
            if len(col) < max_rows:
                col.extend([0] * (max_rows - len(col)))
        # Convertir a matriz numpy (transpuesta)
        matrix = np.array(self.columns).T if self.columns else np.array([[]])
        return matrix

    def get_elements(self) -> List[Dict]:
        """
        Obtiene una lista de elementos válidos (1 o 2) con sus posiciones.

        Returns:
        - elements: List of dicts con información de cada elemento.
        """
        elements = []
        if self.matrix.size == 0:
            return elements
        rows, cols = self.matrix.shape
        for r in range(rows):
            for c in range(cols):
                val = self.matrix[r, c]
                if val in [1, 2]:
                    elements.append({'id': f"{r}_{c}", 'row': r, 'col': c, 'value': val})
        return elements

    def find_possible_pairs(self) -> List[Tuple[str, str]]:
        """
        Encuentra todos los pares de elementos en columnas contiguas que suman 3.

        Returns:
        - possible_pairs: List of tuples, cada tupla contiene dos IDs de elementos que suman 3.
        """
        possible_pairs = []
        # Crear diccionario para acceso rápido por columna
        col_dict = {c: [] for c in range(self.num_columns)}
        for el in self.elements:
            col_dict[el['col']].append(el)
        
        # Iterar sobre columnas contiguas
        for c in range(self.num_columns - 1):
            for el1 in col_dict[c]:
                for el2 in col_dict[c + 1]:
                    if el1['value'] + el2['value'] == 3:
                        possible_pairs.append((el1['id'], el2['id']))
        return possible_pairs

    def build_graph(self) -> nx.Graph:
        """
        Construye un grafo no dirigido a partir de los pares posibles.

        Returns:
        - G: networkx.Graph, grafo con los pares como aristas.
        """
        G = nx.Graph()
        G.add_edges_from(self.possible_pairs)
        return G

    def find_max_matching(self) -> set:
        """
        Encuentra el emparejamiento máximo en el grafo.

        Returns:
        - matching: set of tuples, cada tupla contiene dos IDs de elementos emparejados.
        """
        matching = nx.max_weight_matching(self.graph, maxcardinality=True)
        return matching

    def update_matrix(self) -> np.ndarray:
        """
        Actualiza la matriz estableciendo a 0 los elementos que forman parte de los emparejamientos.

        Returns:
        - updated_matrix: numpy.ndarray, matriz actualizada.
        """
        updated_matrix = self.matrix.copy()
        for pair in self.matching:
            for node in pair:
                r, c = map(int, node.split('_'))
                updated_matrix[r, c] = 0
        return updated_matrix

    def get_groups_info(self) -> List[List[Tuple[int, int]]]:
        """
        Obtiene información detallada de los grupos seleccionados.

        Returns:
        - groups: List of groups, cada grupo es una lista de dos tuplas (Fila, Columna).
        """
        groups = []
        for pair in self.matching:
            group = []
            for node in pair:
                r, c = map(int, node.split('_'))
                group.append((r + 1, c + 1))  # 1-based indexing
            groups.append(group)
        return groups

    def get_elements_count(self) -> Tuple[int, int]:
        """
        Obtiene la cantidad total de elementos y los cubiertos por emparejamientos.

        Returns:
        - total_elements: int, total de elementos (1s y 2s).
        - covered_elements: int, cantidad de elementos cubiertos por emparejamientos.
        """
        total_elements = len(self.elements)
        covered_elements = len(self.matching) * 2
        return total_elements, covered_elements

    def get_groups_count(self) -> int:
        """
        Obtiene la cantidad de grupos utilizados en el emparejamiento.

        Returns:
        - count: int, número de grupos.
        """
        return len(self.groups)

    def get_matrix_as_dataframe(self, updated=False) -> pd.DataFrame:
        """
        Convierte la matriz a un DataFrame de pandas para una visualización más sencilla.

        Parameters:
        - updated: bool, si es True, se usa la matriz actualizada.

        Returns:
        - df: pandas.DataFrame, representación de la matriz.
        """
        mat = self.updated_matrix if updated else self.matrix
        if mat.size == 0:
            return pd.DataFrame()
        df = pd.DataFrame(mat, columns=[f"Columna {c+1}" for c in range(mat.shape[1])],
                          index=[f"Fila {r+1}" for r in range(mat.shape[0])])
        return df

    def plot_matrix(self, updated=False, groups: List[List[Tuple[int, int]]] = None, title="Matriz") -> go.Figure:
        """
        Genera una figura de Plotly para visualizar la matriz y resaltar los grupos seleccionados.

        Parameters:
        - updated: bool, si es True, se usa la matriz actualizada.
        - groups: list of groups, cada grupo es una lista de dos tuplas (Fila, Columna).
        - title: str, título del gráfico.

        Returns:
        - fig: plotly.graph_objects.Figure, figura de la matriz.
        """
        if self.matrix.size == 0:
            return go.Figure()

        mat = self.updated_matrix if updated else self.matrix
        rows, cols = mat.shape
        fig = go.Figure()

        # Añadir Heatmap
        fig.add_trace(go.Heatmap(
            z=mat,
            x=[c + 1 for c in range(cols)],
            y=[r + 1 for r in range(rows)],
            colorscale=[[0, 'white'], [0.5, 'lightblue'], [1, 'lightgreen']],
            showscale=False,
            hovertemplate='Fila %{y}, Col %{x}<br>Valor: %{z}<extra></extra>'
        ))

        # Añadir anotaciones
        annotations = []
        for r in range(rows):
            for c in range(cols):
                val = mat[r, c]
                if val != 0:
                    annotations.append(dict(
                        x=c + 1,
                        y=r + 1,
                        text=str(val),
                        showarrow=False,
                        font=dict(color='black', size=12)
                    ))
        fig.update_layout(
            annotations=annotations
        )

        # Resaltar grupos
        if groups:
            for group in groups:
                (r1, c1), (r2, c2) = group
                # Dibujar línea entre los grupos
                fig.add_shape(
                    type="line",
                    x0=c1,
                    y0=r1,
                    x1=c2,
                    y1=r2,
                    line=dict(color="red", width=2)
                )
                # Resaltar celdas con borde rojo
                for (r, c) in group:
                    fig.add_trace(go.Scatter(
                        x=[c],
                        y=[r],
                        mode='markers',
                        marker=dict(size=40, color='rgba(255,0,0,0)',
                                    line=dict(color='red', width=2)),
                        showlegend=False,
                        hoverinfo='none'
                    ))

        # Configurar diseño
        fig.update_layout(
            title=title,
            xaxis=dict(title='Columnas', tickmode='linear'),
            yaxis=dict(title='Filas', autorange='reversed', tickmode='linear'),
            width=max(600, cols * 60),
            height=max(800, rows * 60)
        )

        return fig
