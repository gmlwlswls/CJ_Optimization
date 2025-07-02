import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set

@dataclass
class WarehouseParameters:
    picking_time: float
    walking_time: float
    cart_capacity: int
    rack_capacity: int
    number_pickers: int

class WarehouseSolver:
    def __init__(self, orders: pd.DataFrame, parameters: pd.DataFrame, od_matrix: pd.DataFrame):
        self.orders = orders.copy()
        self.params = self._load_parameters(parameters)
        self.od_matrix = od_matrix
        self.start_location = od_matrix.index[0]
        self.end_location = od_matrix.index[1]
        
        self._initialize_orders()
        self._validate_input()

    def _load_parameters(self, parameters: pd.DataFrame) -> WarehouseParameters:
        get_param = lambda x: parameters.loc[parameters['PARAMETERS'] == x, 'VALUE'].iloc[0]
        return WarehouseParameters(
            picking_time=float(get_param('PT')),
            walking_time=float(get_param('WT')),
            cart_capacity=int(get_param('CAPA')),
            rack_capacity=int(get_param('RK')),
            number_pickers=int(get_param('PK'))
        )

    def _initialize_orders(self) -> None:
        self.orders['LOC'] = pd.NA
        self.orders['LOC'] = self.orders['LOC'].astype(str)
        self.orders['CART_NO'] = pd.NA
        self.orders['SEQ'] = pd.NA

    def _validate_input(self) -> None:
        if self.orders.empty or self.od_matrix.empty:
            raise ValueError("Input data or OD matrix is empty")
        required_columns = {'ORD_NO', 'SKU_CD'}
        if not required_columns.issubset(self.orders.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(self.orders.columns)}")

    def solve_storage_location(self) -> None:
        """
        Solve Storage Location Assignment Problem (SLAP)
        1단계: 고빈도 SKU 80개 → 입출고지점에 가까운 랙
        2단계: 고빈도 SKU와 자주 주문되는 SKU 80개 → 고빈도 SKU 랙 근처(closeness 기준)
        3단계: 두 개 이상 SKU 주문건 중 자주 등장하는 SKU 80개 → betweeness 기준
        4단계: 나머지 SKU들 → 주문 빈도순 + closeness 기준
        """

        # 초기 변수 설정
        rack_locations = list(self.od_matrix.index[2:])  # oWP_Start, oWP_End 제외
        sku_freq = self.orders['SKU_CD'].value_counts()  # SKU 빈도
        used_racks = set()
        assigned_skus = {}
        rack_capacity = self.params.rack_capacity
        total_racks = len(rack_locations)

        # 1단계: 고빈도 SKU 80개 → 입출고지점 가까운 랙
        high_freq_skus = sku_freq.head(80).index.tolist()

        # 입출고지점과 가까운 랙 순 (closeness 기준)
        from networkx import Graph
        import networkx as nx

        G = nx.from_pandas_adjacency(self.od_matrix.astype(float))
        closeness = nx.closeness_centrality(G, distance='weight')
        sorted_racks = sorted([r for r in rack_locations], key=lambda x: -closeness[x])  # 가까운 순 (값이 클수록 중심)

        rack_iter = iter(sorted_racks)
        for i in range(0, 80, rack_capacity):
            rack = next(rack_iter)
            assigned_skus[rack] = high_freq_skus[i:i + rack_capacity]
            used_racks.add(rack)

        # 2단계: 고빈도 SKU와 함께 주문된 SKU 80개 (고빈도 SKU 제외)
        from collections import Counter
        co_occur = Counter()

        for ord_no, group in self.orders.groupby('ORD_NO'):
            skus = group['SKU_CD'].tolist()
            if any(sku in high_freq_skus for sku in skus):
                for sku in skus:
                    if sku not in high_freq_skus:
                        co_occur[sku] += 1

        related_skus = [sku for sku, _ in co_occur.most_common() if sku not in high_freq_skus][:80]

        # 고빈도 SKU가 배치된 랙 기준으로 closeness 높은 랙에 배치
        rack_left = [r for r in rack_locations if r not in used_racks]
        related_rack_scores = []
        for r in rack_left:
            # 가까운 고빈도 랙과의 거리 합 기준
            score = sum(nx.shortest_path_length(G, source=r, target=hr, weight='weight')
                        for hr in used_racks if nx.has_path(G, r, hr))
            related_rack_scores.append((r, score))

        sorted_related_racks = [r for r, _ in sorted(related_rack_scores, key=lambda x: x[1])]
        rack_iter = iter(sorted_related_racks)

        for i in range(0, len(related_skus), rack_capacity):
            rack = next(rack_iter)
            assigned_skus[rack] = related_skus[i:i + rack_capacity]
            used_racks.add(rack)

        # 3단계: 2개 이상 SKU 주문 건에서 자주 등장하는 SKU 80개 → betweeness 기준
        pair_orders = self.orders.groupby('ORD_NO').filter(lambda x: len(x) > 1)
        pair_sku_freq = pair_orders['SKU_CD'].value_counts()
        middle_skus = [sku for sku in pair_sku_freq.index
                      if sku not in high_freq_skus and sku not in related_skus][:80]

        betweenness = nx.betweenness_centrality(G, weight='weight')
        betweenness_racks = sorted([r for r in rack_locations if r not in used_racks],
                                    key=lambda x: -betweenness[x])
        rack_iter = iter(betweenness_racks)

        for i in range(0, len(middle_skus), rack_capacity):
            rack = next(rack_iter)
            assigned_skus[rack] = middle_skus[i:i + rack_capacity]
            used_racks.add(rack)

        # 4단계: 나머지 SKU들 → 주문 빈도순 + 남은 랙의 closeness 기준
        remaining_skus = [sku for sku in sku_freq.index
                          if sku not in high_freq_skus and sku not in related_skus and sku not in middle_skus]

        closeness_racks = sorted([r for r in rack_locations if r not in used_racks],
                                  key=lambda x: -closeness[x])
        rack_iter = iter(closeness_racks)

        for i in range(0, len(remaining_skus), rack_capacity):
            try:
                rack = next(rack_iter)
                assigned_skus[rack] = remaining_skus[i:i + rack_capacity]
            except StopIteration:
                break  # 랙이 더 이상 없을 경우

        # SKU -> LOC 매핑
        sku_to_location = {}
        for rack, skus in assigned_skus.items():
            for sku in skus:
                sku_to_location[sku] = rack

        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)

    def solve_order_batching(self) -> None:
        """Solve Order Batching and Sequencing Problem (OBSP) using FIFO strategy"""
        unique_orders = sorted(self.orders['ORD_NO'].unique())
        num_carts = len(unique_orders) // self.params.cart_capacity + 1
        
        order_to_cart = {}
        for cart_no in range(1, num_carts + 1):
            start_idx = (cart_no - 1) * self.params.cart_capacity
            end_idx = start_idx + self.params.cart_capacity
            cart_orders = unique_orders[start_idx:end_idx]
            for order in cart_orders:
                order_to_cart[order] = cart_no

        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

    def solve_picker_routing(self) -> None:
        """Solve Pick Routing Problem (PRP) using simple sequencing"""
        self.orders = self.orders.sort_values(['CART_NO', 'LOC'])
        self.orders['SEQ'] = self.orders.groupby('CART_NO').cumcount() + 1

    def solve(self) -> pd.DataFrame:
        """Execute complete warehouse optimization solution"""
        self.solve_storage_location()
        self.solve_order_batching()
        self.solve_picker_routing()
        return self.orders

def main(INPUT: pd.DataFrame, PARAMETER: pd.DataFrame, OD_MATRIX: pd.DataFrame) -> pd.DataFrame:
    solver = WarehouseSolver(INPUT, PARAMETER, OD_MATRIX)
    return solver.solve()

if __name__ == "__main__":
    try:
        test_INPUT = pd.read_csv("Sample_InputData.csv")
        test_PARAM = pd.read_csv("Sample_Parameters.csv")
        test_OD = pd.read_csv("Sample_OD_Matrix.csv", index_col=0, header=0)

        print("Data loaded successfully:")
        print(f"- Orders: {test_INPUT.shape}")
        print(f"- Parameters: {test_PARAM.shape}")
        print(f"- OD Matrix: {test_OD.shape}")

        result = main(test_INPUT, test_PARAM, test_OD)
        result.to_csv("Sample_OutputData.csv", index=False)
        print("\nOptimization completed. Results preview:")
        print(result.head())

    except FileNotFoundError as e:
        print(f"Error: Unable to load required files - {str(e)}")
    except (pd.errors.DataError, pd.errors.EmptyDataError) as e:
        print(f"Error: Data validation failed - {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")