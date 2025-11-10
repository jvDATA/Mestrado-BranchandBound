import sys
import math
import os
import numpy as np

class Data:
 

    def __init__(self, instance_name):
        """
        Construtor da classe. Inicializa os atributos.
        """
        if not os.path.exists(instance_name):
            raise FileNotFoundError(f"Erro: Arquivo não encontrado em '{instance_name}'")
            
        self.instance_name = instance_name
        self.dimension = -1
        self.dist_matrix = None
        self.x_coord = None
        self.y_coord = None
        self.explicit_coord = False
        
       
        self.INFINITE = float('inf')

    def read(self):
        
        with open(self.instance_name, 'r') as f:
            # Iterator
            words = (word for line in f for word in line.strip().split())

            params = {}
            current_word = next(words, None)
            
            while current_word not in ("NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "EOF"):
                ### SE N FUNCIONAR VAI FICAR BUGADO E FÉ
               
                key = current_word.rstrip(':')
                
                # Pega a próxima palavra
                next_val = next(words, None)

                # Verifica se estamos no formato "CHAVE : VALOR"
                if next_val == ':':
                    
                    value = next(words, None)
                else:
                    # Não, então o 'next_val' JÁ ERA o valor (formato "CHAVE: VALOR")
                    value = next_val
                
                # Armazena a chave e o valor
                params[key] = value
                
                # Pega a próxima palavra para continuar o loop
                current_word = next(words, None)
            
            self.dimension = int(params.get("DIMENSION", 0))
            edge_weight_type = params.get("EDGE_WEIGHT_TYPE", "")
            edge_weight_format = params.get("EDGE_WEIGHT_FORMAT", "")
            ### DEBUG
            print(f"Dimension: {self.dimension}, Edge Weight Type: {edge_weight_type}, Edge Weight Format: {edge_weight_format}")
            # Aloca as estruturas de dados usando NumPy para eficiência
            self.dist_matrix = np.full((self.dimension, self.dimension), self.INFINITE, dtype=float)
            self.x_coord = np.zeros(self.dimension, dtype=float)
            self.y_coord = np.zeros(self.dimension, dtype=float)

            # 2. Leitura dos dados (coordenadas ou matriz de adjacência)
            if edge_weight_type in ("EUC_2D", "CEIL_2D", "GEO", "ATT"):
                self.explicit_coord = True
                
                # Pula para a seção de coordenadas
                for i in range(self.dimension):
                    # Ignora o ID do nó
                    next(words) 
                    self.x_coord[i] = float(next(words))
                    self.y_coord[i] = float(next(words))

                # Calcula a matriz de distância com base no tipo
                if edge_weight_type == "EUC_2D":
                    for i in range(self.dimension):
                        for j in range(self.dimension):
                            if i != j:
                                self.dist_matrix[i, j] = math.floor(self._calc_dist_euc(i, j) + 0.5)
                            else:
                                self.dist_matrix[i, j] = self.INFINITE
                
                elif edge_weight_type == "CEIL_2D":
                    for i in range(self.dimension):
                        for j in range(self.dimension):
                            if i != j:
                                self.dist_matrix[i, j] = math.ceil(self._calc_dist_euc(i, j))
                            else:
                                self.dist_matrix[i, j] = self.INFINITE

                elif edge_weight_type == "GEO":
                    lat, lon = self._calc_lat_long()
                    for i in range(self.dimension):
                        for j in range(self.dimension):
                            if i != j:
                                self.dist_matrix[i, j] = self._calc_dist_geo(lat, lon, i, j)
                            else:
                                self.dist_matrix[i, j] = self.INFINITE

                elif edge_weight_type == "ATT":
                    for i in range(self.dimension):
                        for j in range(self.dimension):
                            if i != j:
                                self.dist_matrix[i, j] = self._calc_dist_att(i, j)
                            else:
                                self.dist_matrix[i, j] = self.INFINITE

            elif edge_weight_type == "EXPLICIT":
                # ajuste dos pesos explícitos
                ### FALTA TER IMPLEMENTAÇÃO PARA OUTROS FORMATOS
                weights = [float(w) for w in words if w != 'EOF']
                k = 0

                if edge_weight_format == "FULL_MATRIX":
                    for i in range(self.dimension):
                        for j in range(self.dimension):
                            if i != j:
                                self.dist_matrix[i, j] = weights[k]
                            else:
                                self.dist_matrix[i, j] = self.INFINITE
                            k += 1
                
                elif edge_weight_format == "UPPER_ROW":
                    for i in range(self.dimension):
                        for j in range(i + 1, self.dimension):
                            self.dist_matrix[i, j] = self.dist_matrix[j, i] = weights[k]
                            k += 1
                    for i in range(self.dimension):
                        self.dist_matrix[i, i] = self.INFINITE

                elif edge_weight_format == "LOWER_ROW":
                    for i in range(self.dimension):
                        for j in range(0, i):
                            self.dist_matrix[i, j] = self.dist_matrix[j, i] = weights[k]
                            k += 1
                   
                    for i in range(self.dimension):
                        self.dist_matrix[i, i] = self.INFINITE
                

                elif edge_weight_format == "UPPER_DIAG_ROW":
                    for i in range(self.dimension):
                        for j in range(i, self.dimension):
                            val = weights[k]
                            if i != j:
                                self.dist_matrix[i, j] = self.dist_matrix[j, i] = val
                            else:
                                self.dist_matrix[i, i] = self.INFINITE
                            k += 1
                
                elif edge_weight_format == "LOWER_DIAG_ROW":
                    for i in range(self.dimension):
                        for j in range(0, i + 1):
                            val = weights[k]
                            if i != j:
                                self.dist_matrix[i, j] = self.dist_matrix[j, i] = val
                            else:
                                self.dist_matrix[i, i] = self.INFINITE
                            k += 1

                elif edge_weight_format == "UPPER_COL":
                    for j in range(self.dimension):
                        for i in range(0, j):
                            self.dist_matrix[i, j] = self.dist_matrix[j, i] = weights[k]
                            k += 1
                    for i in range(self.dimension):
                        self.dist_matrix[i, i] = self.INFINITE


                elif edge_weight_format == "LOWER_COL":
                    for j in range(self.dimension):
                        for i in range(j + 1, self.dimension):
                            self.dist_matrix[i, j] = self.dist_matrix[j, i] = weights[k]
                            k += 1
                    for i in range(self.dimension):
                        self.dist_matrix[i, i] = self.INFINITE

                elif edge_weight_format == "UPPER_DIAG_COL":
                    for j in range(self.dimension):
                        for i in range(0, j + 1):
                            val = weights[k]
                            if i != j:
                                self.dist_matrix[i, j] = self.dist_matrix[j, i] = val
                            else:
                                self.dist_matrix[i, i] = self.INFINITE
                            k += 1
                  
                elif edge_weight_format == "LOWER_DIAG_COL":
                    for j in range(self.dimension):
                        for i in range(j, self.dimension):
                            val = weights[k]
                            if i != j:
                                self.dist_matrix[i, j] = self.dist_matrix[j, i] = val
                            else:
                                self.dist_matrix[i, i] = self.INFINITE
                            k += 1
                

                else:
                    raise NotImplementedError(f"Formato de peso '{edge_weight_format}' não suportado.")
            
            else:
                 raise NotImplementedError(f"Tipo de peso '{edge_weight_type}' não suportado.")

    ### Métodos de Cálculo  

    def _calc_dist_euc(self, i, j):
        return math.sqrt((self.x_coord[i] - self.x_coord[j])**2 + (self.y_coord[i] - self.y_coord[j])**2)

    def _calc_dist_att(self, i, j):
        rij = math.sqrt(((self.x_coord[i] - self.x_coord[j])**2 + (self.y_coord[i] - self.y_coord[j])**2) / 10.0)
        tij = math.floor(rij + 0.5)
        return tij + 1 if tij < rij else tij

    def _calc_lat_long(self):
        PI = 3.141592
        latitude = np.zeros(self.dimension)
        longitude = np.zeros(self.dimension)
        for i in range(self.dimension):
            deg_x = int(self.x_coord[i])
            min_x = self.x_coord[i] - deg_x
            latitude[i] = PI * (deg_x + 5.0 * min_x / 3.0) / 180.0
            
            deg_y = int(self.y_coord[i])
            min_y = self.y_coord[i] - deg_y
            longitude[i] = PI * (deg_y + 5.0 * min_y / 3.0) / 180.0
        return latitude, longitude

    def _calc_dist_geo(self, lat, lon, i, j):
        RRR = 6378.388
        q1 = math.cos(lon[i] - lon[j])
        q2 = math.cos(lat[i] - lat[j])
        q3 = math.cos(lat[i] + lat[j])
        return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    # --- Métodos Públicos  ---
    def get_dimension(self):
        return self.dimension

    def get_distance(self, i, j):
        
        
        return self.dist_matrix[i - 1, j - 1]

    def get_dist_matrix(self):
        return self.dist_matrix

    def print_dist_matrix(self):
        for i in range(1, self.dimension + 1):
            for j in range(1, self.dimension + 1):
                dist = self.get_distance(i, j)
                print(f"{dist:<10.2f}", end="")
            print()


if __name__ == "__main__":
    # Verifica se o nome do arquivo foi passado como argumento
    if len(sys.argv) < 2:
        print("Uso: python data.py [caminho_para_instancia]")
        sys.exit(1)

    instance_file = sys.argv[1]

    try:
        # Cria o objeto Data e lê o arquivo
        data = Data(instance_file)
        data.read()

        n = data.get_dimension()
        print(f"Dimension: {n}")
        print("Distance Matrix:")
        data.print_dist_matrix()

        # Exemplo de cálculo de custo para uma solução simples (1 -> 2 -> ... -> n -> 1)
        print("\nExemplo de Solucao s = ", end="")
        cost = 0.0
        tour = list(range(1, n + 1))
        
        for i in range(n - 1):
            u, v = tour[i], tour[i+1]
            print(f"{u} -> ", end="")
            cost += data.get_distance(u, v)
        
        # Fecha o ciclo
        last_node, first_node = tour[-1], tour[0]
        cost += data.get_distance(last_node, first_node)
        print(f"{last_node} -> {first_node}")

        print(f"Custo de S: {cost}")

    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        print(e)
        sys.exit(1)