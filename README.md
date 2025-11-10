# Branch and Bound para TSP

Este projeto implementa um algoritmo Branch and Bound para resolver o Problema do Caixeiro Viajante (TSP - Traveling Salesman Problem).

## Requisitos

- Python 3.x
- pip (gerenciador de pacotes Python)

## Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/[seu-usuario]/Mestrado-BranchandBound.git
cd Mestrado-BranchandBound
```

2. Crie um ambiente virtual:
```bash
python -m venv .venv
```

3. Ative o ambiente virtual:

No Windows:
```bash
.\venv\Scripts\activate
```

No Linux/macOS:
```bash
source venv/bin/activate
```

4. Instale as dependências:
```bash
pip install -r Requirements.txt
```

## Como Executar

Para executar o algoritmo, use o arquivo `main.py` especificando o caminho para uma instância TSP:

```bash
python main.py instances/[nome-da-instancia].tsp
```

Exemplo:
```bash
python main.py instances/berlin52.tsp
```

## Instâncias Disponíveis

O projeto inclui várias instâncias TSP na pasta `instances/`, incluindo:
- berlin52.tsp (52 cidades)
- att48.tsp (48 cidades)
- eil51.tsp (51 cidades)
- E muitas outras...

Escolha a instância adequada ao seu caso de teste. Instâncias maiores requerem mais tempo de processamento.

## Estrutura do Projeto

- `main.py`: Arquivo principal de execução
- `data.py`: Módulo para manipulação de dados
- `instances/`: Diretório contendo as instâncias TSP
- `Requirements.txt`: Lista de dependências Python
