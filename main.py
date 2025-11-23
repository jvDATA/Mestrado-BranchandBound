import sys
import math
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from queue import PriorityQueue
import time

from data import Data


class BranchAndBoundSolver:
    """
    Resolve o Problema do Caixeiro Viajante usando Branch and Bound
    com estratégia Best Bound (best-first search) ou DFS.
    """
    def __init__(self, data_instance):
        self.data = data_instance
        self.dimension = data_instance.get_dimension()
        self.root_cost_matrix = data_instance.get_dist_matrix().copy()
        self.primal_bound = float('inf') 
        self.dual_bound = float('-inf') 
        self.best_solution = []
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def _find_subtours(self, assignment, n):
        """
        Encontra todos os subtours em uma solução de designação.
        
        Args:
            assignment: array onde assignment[i] indica a cidade destino da cidade i
            n: número de cidades
            
        Returns:
            lista de subtours (cada subtour é uma lista de cidades)
        """
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
        
        return subtours
    
    def _calculate_lower_bound(self, node):
        """
        Calcula o limite inferior para um nó usando o Algoritmo Húngaro.
        Também identifica subtours e verifica a viabilidade da solução.
        
        Args:
            node: dicionário representando o nó da árvore de busca
        """
        # Resolve o problema de atribuição usando o algoritmo húngaro
        row_ind, col_ind = linear_sum_assignment(node["cost_matrix"])
        
        # Calcula o custo total da atribuição
        node["lower_bound"] = node["cost_matrix"][row_ind, col_ind].sum()
        
        # Encontra os subtours resultantes
        node["subtours"] = self._find_subtours(col_ind, self.dimension)

        # Verifica a viabilidade e encontra o menor subtour
        if len(node["subtours"]) == 1:
            # Solução viável: um único tour visitando todas as cidades
            node["is_feasible"] = True
            node["smallest_subtour_idx"] = 0
        else:
            # Solução inviável: múltiplos subtours
            node["is_feasible"] = False
            # Encontra o índice do menor subtour para fazer branching
            node["smallest_subtour_idx"] = np.argmin([len(st) for st in node["subtours"]])
    
    def solve(self, search_strategy="best_bound", verbose=True):
        """
        Inicia e gerencia o processo do Branch and Bound.
        
        Args:
            search_strategy: "dfs" para depth-first search ou 
                           "best_bound" para best-first search (padrão)
            verbose: se True, exibe informações durante a execução
            
        Returns:
            tuple: (custo_ótimo, tour_ótimo)
        """
        start_time = time.time()
        
        # Escolhe a estrutura de dados baseada na estratégia
        if search_strategy == "best_bound":
            # Fila de prioridade: menor lower bound tem maior prioridade
            pq = PriorityQueue()
            counter = 0  # Contador para desempate
        else:
            # Pilha para DFS
            stack = []

        # 1. Criar o nó raiz
        root_node = {
            "forbidden_arcs": set(),
            "cost_matrix": self.root_cost_matrix.copy(),
            "lower_bound": 0,
            "subtours": [],
            "smallest_subtour_idx": -1,
            "is_feasible": False,
        }

        # Calcula o primeiro limite inferior para o nó raiz
        self._calculate_lower_bound(root_node)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Branch and Bound - Estratégia: {search_strategy.upper()}")
            print(f"{'='*70}")
            print(f"Dimensão do problema: {self.dimension} cidades")
            print(f"Lower bound inicial (nó raiz): {root_node['lower_bound']:.2f}")
            print(f"Número de subtours no nó raiz: {len(root_node['subtours'])}")
            print(f"{'='*70}\n")
        
        # Adiciona o nó raiz na estrutura apropriada
        if search_strategy == "best_bound":
            pq.put((root_node["lower_bound"], counter, root_node))
            counter += 1
        else:
            stack.append(root_node)

        # 2. Loop principal do B&B
        iteration = 0
        last_print_time = start_time
        
        while (not pq.empty() if search_strategy == "best_bound" else len(stack) > 0):
            self.nodes_explored += 1
            iteration += 1
            
            # Seleciona o próximo nó baseado na estratégia
            if search_strategy == "best_bound":
                _, _, current_node = pq.get()
            else:  # DFS
                current_node = stack.pop()

            current_cost = current_node["lower_bound"]

            # Poda por limite (bounding)
            if current_cost >= self.primal_bound:
                self.nodes_pruned += 1
                continue

            # Verifica se encontrou uma solução viável
            if current_node["is_feasible"]:
                if verbose:
                    print(f"✓ Solução viável encontrada!")
                    print(f"  Custo: {current_cost:.2f}")
                    print(f"  Tour: {' → '.join(map(str, current_node['subtours'][0]))} → {current_node['subtours'][0][0]}")
                
                if current_cost < self.primal_bound:
                    if verbose:
                        print(f"  ★ NOVA MELHOR SOLUÇÃO! (anterior: {self.primal_bound:.2f})")
                    self.primal_bound = current_cost
                    self.best_solution = current_node["subtours"][0].copy()
                
                if verbose:
                    print()
            else:
                # Branching: ramifica no menor subtour
                subtour_to_branch = current_node["subtours"][current_node["smallest_subtour_idx"]]
                
                # Print periódico do progresso
                current_time = time.time()
                if verbose and (current_time - last_print_time >= 2.0 or iteration <= 10):
                    queue_size = pq.qsize() if search_strategy == "best_bound" else len(stack)
                    elapsed = current_time - start_time
                    best_str = f"{self.primal_bound:.2f}" if self.primal_bound != float('inf') else "∞"
                    print(f"[{elapsed:.1f}s] Nós: {self.nodes_explored:6d} | "
                          f"Fila: {queue_size:6d} | "
                          f"Melhor: {best_str:>10} | "
                          f"LB atual: {current_cost:.2f}")
                    last_print_time = current_time
                
                # Cria um nó filho para cada arco do subtour
                for i in range(len(subtour_to_branch)):
                    from_node = subtour_to_branch[i]
                    to_node = subtour_to_branch[(i + 1) % len(subtour_to_branch)]
                    
                    new_forbidden_arc = (from_node, to_node)
                    
                    # Cria o novo nó filho
                    child_node = {
                        "forbidden_arcs": current_node["forbidden_arcs"] | {new_forbidden_arc},
                        "cost_matrix": current_node["cost_matrix"].copy(),
                        "lower_bound": 0,
                        "subtours": [],
                        "smallest_subtour_idx": -1,
                        "is_feasible": False,
                    }

                    # Proíbe o arco na matriz de custo do filho
                    child_node["cost_matrix"][from_node, to_node] = float('inf')

                    # Calcula o lower bound do filho
                    self._calculate_lower_bound(child_node)

                    # Adiciona o filho na estrutura se for promissor
                    if child_node["lower_bound"] < self.primal_bound:
                        if search_strategy == "best_bound":
                            pq.put((child_node["lower_bound"], counter, child_node))
                            counter += 1
                        else:
                            stack.append(child_node)
                    else:
                        self.nodes_pruned += 1
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RESULTADO FINAL")
            print(f"{'='*70}")
            print(f"Estratégia: {search_strategy.upper()}")
            print(f"Custo ótimo: {self.primal_bound:.2f}")
            if self.best_solution:
                print(f"Tour ótimo: {' → '.join(map(str, self.best_solution))} → {self.best_solution[0]}")
            print(f"\nEstatísticas:")
            print(f"  • Nós explorados: {self.nodes_explored:,}")
            print(f"  • Nós podados: {self.nodes_pruned:,}")
            print(f"  • Tempo de execução: {elapsed_time:.2f} segundos")
            print(f"  • Nós por segundo: {self.nodes_explored/elapsed_time:.0f}")
            print(f"{'='*70}\n")
        
        return self.primal_bound, self.best_solution


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python branch_and_bound.py <caminho_instancia> [estrategia]")
        print("\nParâmetros:")
        print("  caminho_instancia: caminho para o arquivo da instância TSP")
        print("  estrategia: 'dfs' ou 'best_bound' (padrão: best_bound)")
        print("\nExemplo:")
        print("  python branch_and_bound.py instances/burma14.tsp best_bound")
        sys.exit(1)

    instance_file = sys.argv[1]
    search_strategy = sys.argv[2].lower() if len(sys.argv) >= 3 else "best_bound"
    
    # Valida a estratégia
    if search_strategy not in ["dfs", "best_bound"]:
        print(f"Erro: estratégia '{search_strategy}' inválida.")
        print("Use 'dfs' ou 'best_bound'")
        sys.exit(1)
    
    try:
        # 1. Lê a instância usando a classe Data
        data = Data(instance_file)
        data.read()
        print(f"\nInstância '{os.path.basename(instance_file)}' carregada com sucesso.")
        print(f"Dimensão: {data.get_dimension()} cidades")

        # 2. Cria e roda o solver
        solver = BranchAndBoundSolver(data)
        optimal_cost, optimal_tour = solver.solve(search_strategy=search_strategy, verbose=True)

        # 3. Exibe resultados finais
        if optimal_cost != float('inf'):
            print("✓ Solução ótima encontrada com sucesso!")
        else:
            print("✗ Não foi possível encontrar uma solução.")

    except FileNotFoundError:
        print(f"\nERRO: Arquivo '{instance_file}' não encontrado.")
        sys.exit(1)
    except (ValueError, NotImplementedError) as e:
        print(f"\nERRO: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExecução interrompida pelo usuário.")
        print(f"Nós explorados até o momento: {solver.nodes_explored}")
        if solver.primal_bound != float('inf'):
            print(f"Melhor solução encontrada: {solver.primal_bound:.2f}")
        sys.exit(0)