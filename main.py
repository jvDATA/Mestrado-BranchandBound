import sys
import math
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

from data import Data


class BranchAndBoundSolver:
    """
    Resolve o Problema do Caixeiro Viajante usando Branch and Bound
    com estratégia de busca em profundidade (DFS).
    """
    def __init__(self, data_instance):
        self.data = data_instance
        self.dimension = data_instance.get_dimension()
        self.root_cost_matrix = data_instance.get_dist_matrix().copy()
        self.primal_bound = float('inf') 
        self.dual_bound = float('-inf') 
        self.best_solution = []

    def _find_subtours(self, assignment, n):
        """
        Encontra todos os subtours em uma solução de designação.
        
        """
        ### implementei essa bomba errada. Num é possivelzzz
        # print(f"Assignment: {assignment}")
        visited = [False] * n
        subtours = []
        for i in range(n):
            if not visited[i]:
                current_tour = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    current_tour.append(j) 
                    j = assignment[j]
                subtours.append(current_tour)
        ### DEBUG
        # print(f"Subtours encontrados: {len(subtours)}")

        return subtours
    
    def solve(self):
        """
        Inicia e gerencia o processo do Branch and Bound.
        """
        
        stack = [] 

        # 1. Criar o nó raiz
        root_node = {
            "forbidden_arcs": set(),
            "cost_matrix": self.root_cost_matrix,
            "lower_bound": 0,
            "subtours": [],
            "smallest_subtour_idx": -1,
            "is_feasible": False,
        }

        # Calcula o primeiro limite inferior para o nó raiz
        self._calculate_lower_bound(root_node)
        
       
        stack.append(root_node)

        # 2. Loop principal do B&B (DFS)
        while stack:
            # print(len(stack))
            
            current_node = stack.pop() 

            current_cost = current_node["lower_bound"]

            ###  Poda (limite) 
            if current_cost >= self.primal_bound:
    
                continue

            if current_node["is_feasible"]:
                
                print(f"Solução viável encontrada com custo: {current_cost}")
                if current_cost < self.primal_bound:
                    print(f"Novo melhor Primal Bound: {current_cost}")
                    self.primal_bound = current_cost
                    self.best_solution = sorted(current_node["subtours"][0])
            else:
                # Branching
                # Pega o menor subtour 
                subtour_to_branch = current_node["subtours"][current_node["smallest_subtour_idx"]]
                print(f"Ramificando no subtour: {subtour_to_branch}")
                for i in range(len(subtour_to_branch)):
                    # Para cada arco no subtour, cria um novo nó proibindo esse arco
                    from_node = subtour_to_branch[i]
                    to_node = subtour_to_branch[(i + 1) % len(subtour_to_branch)]
                    
                    new_forbidden_arc = (from_node, to_node)
                    
                    # Cria o novo nó filho
                    child_node = {
                        "forbidden_arcs": current_node["forbidden_arcs"].union({new_forbidden_arc}),
                        "cost_matrix": current_node["cost_matrix"].copy(),
                        "lower_bound": 0,
                        "subtours": [],
                        "smallest_subtour_idx": -1,
                        "is_feasible": False,
                    }

                    # Proíbe o arco na matriz de custo do filho
                    child_node["cost_matrix"][from_node, to_node] = float('inf')

                    self._calculate_lower_bound(child_node)

                    # Adiciona o filho na pilha
                    if child_node["lower_bound"] < self.primal_bound:
                        
                        stack.append(child_node)
        return self.primal_bound, self.best_solution


    def _calculate_lower_bound(self, node):
        """
        Calcula o limite inferior para um nó usando o Algoritmo Húngaro.
        """
        
        row_ind, col_ind = linear_sum_assignment(node["cost_matrix"])
        
        # Calcula o custo total e armazena os arcos da solução
        node["lower_bound"] = node["cost_matrix"][row_ind, col_ind].sum()
        
        # Encontra os subtours resultantes
        node["subtours"] = self._find_subtours(col_ind, self.dimension)

        # Verifica a viabilidade e encontra o menor subtour
        if len(node["subtours"]) == 1: #and len(node["subtours"][0]) == self.dimension:
            node["is_feasible"] = True
        else:
            node["is_feasible"] = False
            # Retorna o índice do menor subtour
            node["smallest_subtour_idx"] = np.argmin([len(st) for st in node["subtours"]])
    



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python seu_script.py [caminho_para_instancia]")
        sys.exit(1)

    instance_file = sys.argv[1]
    ### Ta explodindo mt rápido. Vou mudar essa bomba pra julia mesmo
    try:
        # 1. Le a instância usando a classe Data
        data = Data(instance_file)
        data.read()
        print(f"Instância '{os.path.basename(instance_file)}' lida com dimensão {data.get_dimension()}.\n")

        # 2. Cria e roda o solver
        solver = BranchAndBoundSolver(data)
        optimal_cost, optimal_tour = solver.solve()

        # 3. Exibe resultados
        if optimal_cost != float('inf'):
            print("\n" + "="*40)
            print("Solução ótima encontrada!")
            print(f"Custo Ótimo: {optimal_cost}")
            print(f"Tour Ótimo: {' -> '.join(map(str, optimal_tour))} -> {optimal_tour[0]}")
            print("="*40)
        else:
            print("\nNão foi possível encontrar uma solução.")

    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)