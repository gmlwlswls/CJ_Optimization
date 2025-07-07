import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import SpectralClustering

# FIFO로 처리하는 샘플 알고리즘 입니다.
# def main처럼 데이터프레임 타입으로 결과 리턴해주시면 됩니다. 데이터는 제공한 Sample_OutputData.csv와 동일한 형태로 리턴해주시면 됩니다.

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
        #추가한 부분 -> OBSP에 사용하기 위해서(test용으로 한거라 참고만 해도 괜찮음!)
        self.cooc = None
        self.zone_cooc = None, 
        self.ordered_zone = None
        self.rack_to_zone = None
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


#주문 빈도수와 상품 연관성 고려한 SLAP

    def solve_storage_location(self) -> None:
        """Solve Storage Location Assignment Problem (SLAP) using SKU frequency and co-occurrence clustering"""
        from collections import defaultdict
        from itertools import combinations

        # 1. SKU 출고 빈도 계산 (NUM_PCS가 없으면 주문 건수 기준)
        if 'NUM_PCS' in self.orders.columns:
            freq = self.orders.groupby('SKU_CD')['NUM_PCS'].sum()
        else:
            freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2. SKU 간 공동 주문 연관성(co-occurrence) 계산
        cooc = defaultdict(int)
        for _, group in self.orders.groupby('ORD_NO'):
            for a, b in combinations(group['SKU_CD'], 2):
                cooc[(a, b)] += 1
                cooc[(b, a)] += 1
        self.cooc = cooc
        # 3. 시작 지점에서 가까운 랙부터 정렬
        rack_locations = self.od_matrix.index[2:]
        dist_start = self.od_matrix.loc[self.start_location, rack_locations]
        rack_sorted = dist_start.sort_values().index.tolist()

        # 4. 그리디 클러스터링: 가장 연관성이 높은 SKU들을 같은 클러스터(랙)로 묶기
        assigned = set()
        clusters = []
        for sku in skus_by_freq:
            if sku in assigned:
                continue
            cluster = {sku}
            candidates = [s for s in skus_by_freq if s not in assigned]
            # 연관성 높은 순 정렬
            candidates.sort(key=lambda x: cooc.get((sku, x), 0), reverse=True)
            for c in candidates:
                if len(cluster) < self.params.rack_capacity:
                    cluster.add(c)
                else:
                    break
            assigned |= cluster
            clusters.append(cluster)

        # 5. 클러스터를 랙에 매핑
        sku_to_location = {}
        for rack, cluster in zip(rack_sorted, clusters):
            for sku in cluster:
                sku_to_location[sku] = rack

        # 6. 할당되지 않은 남은 SKU 처리 (랜덤 또는 빈도 순)
        remaining = [s for s in skus_by_freq if s not in sku_to_location]
        print(len(remaining))
        idx = len(clusters)
        for sku in remaining:
            rack = rack_sorted[idx // self.params.rack_capacity]
            sku_to_location[sku] = rack
            idx += 1
        # 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)

#FIFO(OBSP)
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

#단순 그리드클러스터링 배치에 관한 OBSP(test용)
    '''def solve_order_batching(self) -> None:
        """Solve OBSP: 주문을 카트에 배치하고, 카트별 ZONE 순서를 기반으로 SEQ 지정"""
        from collections import defaultdict

        # 1️⃣ 주문별 SKU 목록
        order_skus = self.orders.groupby('ORD_NO')['SKU_CD'].apply(list).to_dict()
        order_locs = self.orders.groupby('ORD_NO')['LOC'].apply(list).to_dict()

        # 2️⃣ 주문별 Zone (랙) 집합
        order_zones = {
            ord_no: set(order_locs[ord_no])
            for ord_no in order_locs
        }

        order_ids = list(order_skus.keys())

        # 3️⃣ 주문 간 유사도 계산 (SKU 연관성의 합)
        order_sim = defaultdict(float)
        for i in range(len(order_ids)):
            for j in range(i+1, len(order_ids)):
                o1, o2 = order_ids[i], order_ids[j]
                sim = sum(
                    self.cooc.get((s1, s2), 0)
                    for s1 in order_skus[o1]
                    for s2 in order_skus[o2]
                )
                order_sim[(o1, o2)] = sim
                order_sim[(o2, o1)] = sim

        # 4️⃣ 주문별 유사 이웃 정렬
        neighbors = {o: [] for o in order_ids}
        for (o1, o2), sim in order_sim.items():
            neighbors[o1].append((o2, sim))
        for o in neighbors:
            neighbors[o].sort(key=lambda x: -x[1])  # 유사도 내림차순

        # 5️⃣ Greedy Batching
        assigned = set()
        order_to_cart = {}
        cart_no = 1
        for o in order_ids:
            if o in assigned:
                continue
            batch = [o]
            assigned.add(o)
            for cand, _ in neighbors[o]:
                if len(batch) >= self.params.cart_capacity:
                    break
                if cand not in assigned:
                    batch.append(cand)
                    assigned.add(cand)
            for ord_id in batch:
                order_to_cart[ord_id] = cart_no
            cart_no += 1

        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

        # 6️⃣ 카트별 Zone 순서로 SEQ 부여
        # (입출고지점에서 가까운 랙 순으로 카트별 SEQ 지정)
        cart_zones = self.orders.groupby('CART_NO')['LOC'].apply(set).to_dict()
        dist_start = self.od_matrix.loc[self.start_location]

        cart_order = sorted(cart_zones.keys(), key=lambda c: min(
            dist_start[loc] for loc in cart_zones[c] if loc in dist_start
        ))
        cart_to_seq = {cart: idx + 1 for idx, cart in enumerate(cart_order)}

        self.orders['SEQ'] = self.orders['CART_NO'].map(cart_to_seq)

'''

    def solve_picker_routing(self) -> None:
        """Solve Pick Routing Problem (PRP) using simple sequencing"""
        self.orders = self.orders.sort_values(['CART_NO', 'LOC'])
        self.orders['SEQ'] = self.orders.groupby('CART_NO').cumcount() + 1

    def solve(self) -> pd.DataFrame:
        """Execute complete warehouse optimization solution"""
        self.solve_storage_location()
        self.solve_order_batching()
        self.solve_picker_routing()
        if self.orders['LOC'].isna().any():
            raise ValueError("LOC에 할당되지 않은 SKU가 있습니다.")
        if self.orders['CART_NO'].isna().any():
            raise ValueError("CART_NO가 지정되지 않았습니다.")
        if self.orders['SEQ'].isna().any():
            raise ValueError("SEQ가 지정되지 않았습니다.")
        return self.orders

def main(INPUT: pd.DataFrame, PARAMETER: pd.DataFrame, OD_MATRIX: pd.DataFrame) -> pd.DataFrame:
    solver = WarehouseSolver(INPUT, PARAMETER, OD_MATRIX)
    solver.solve().to_csv('./SLAP_LOC.csv', index= False)
    return solver.solve()

if __name__ == "__main__":
    import time
    test_INPUT = pd.read_csv("./data/Sample_InputData.csv")
    test_PARAM = pd.read_csv("./data/Sample_Parameters.csv")
    test_OD = pd.read_csv("./data/Sample_OD_Matrix.csv", index_col=0, header=0)
    start_time = time.time()
    try:
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
    
    print("total_time : ", time.time() - start_time)