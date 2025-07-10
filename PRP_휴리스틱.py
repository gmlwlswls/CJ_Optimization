import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict
from itertools import combinations
from statistics import mode
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from math import ceil

# def main처럼 데이터프레임 타입으로 결과 리턴해주시면 됩니다. 데이터는 제공한 Sample_OutputData.csv와 동일한 형태로 리턴해주시면 됩니다.
# SLAP - 고빈도 우선 + 주문 SKU 유사도 > 그리디 클러스터링으로 입출고 지점과 가까운 순으로 배정
# OLBP - 주문별 SKU유사도 + 랙 위치 유사도 > KMeans 클러스터링으로 4개의 주문씩 클러스터링
# + 작업량 맞추는 작업
# PRP - 2_opt + Simulated Annealing


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

    ##### 1) SLAP
    def solve_storage_location(self) -> None:
        """Solve Storage Location Assignment Problem (SLAP) using SKU frequency and co-occurrence clustering"""

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
        print('할당되지 않은 SKU :', len(remaining))
        idx = len(clusters)
        for sku in remaining:
            rack = rack_sorted[idx // self.params.rack_capacity]
            sku_to_location[sku] = rack
            idx += 1
        # 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)
    
    #####2) OBSP
    def solve_order_batching(self) -> None:
        def extract_loc_num(loc_name) :
            """
            LOC 이름(WP_0001 등)에서 숫자 부분만 추출하여 정수로 반환
            """
            match = re.search(r'\d+', loc_name)
            return int(match.group()) if match else None
                
        def estimate_row_count_from_od_matrix(od_matrix) :
            """
            OD Matrix에서 랙 간 거리 분포를 보고 한 줄에 몇 개 있는지, 전체 줄 수를 추정
            """
            row_len_candidates = []
            for rack in od_matrix.index :  
                distances = od_matrix.loc[rack].sort_values()
                diffs = distances.diff().fillna(0)
                
                # 급격히 증가하는 지점 찾기 (거리 급증)
                jump_indices = np.where(diffs > 5)[0]  # 거리 단위 기준 조정 가능
                
                if len(jump_indices) > 0:
                    row_len_candidates.append(jump_indices[0])  # 첫 번째 점프 위치
                    
            # 최빈값 = 추정된 한 줄 랙 수
            est_rack_count = mode(row_len_candidates) # 한 구역에 포함된 랙의 갯수
            # 전체 랙 수에서 줄 수 계산
            total_racks = len(od_matrix)
            est_row_count = total_racks // est_rack_count # 구역 갯수
            return est_rack_count        

        def assign_zone_by_locnum(slap_loc_df, rack_count):
            """
            LOC_NUM을 기반으로 ZONE을 부여한다.

            Parameters:
            - slap_df: SLAP_LOC DataFrame
            - row_len: 추정된 한 줄에 있는 랙 수

            Returns:
            - ZONE이 부여된 DataFrame
            """
            slap_loc_df = slap_loc_df.copy()

            # LOC에서 숫자 추출
            slap_loc_df['LOC_NUM'] = slap_loc_df['LOC'].map(extract_loc_num)

            # ZONE 부여
            slap_loc_df['ZONE'] = slap_loc_df['LOC_NUM'].apply(
                lambda x: f"ZONE_{(x - 1) // rack_count + 1}" if pd.notnull(x) else None
            )
            
            return slap_loc_df
        
        rack_count = estimate_row_count_from_od_matrix(self.od_matrix)        
        zone_assign_df = assign_zone_by_locnum(self.orders, rack_count)
        
        # 주문-sku 매트릭스 & 주문-ZONE 매트릭스
        order_sku_matrix = zone_assign_df.pivot_table(index='ORD_NO', columns='SKU_CD', aggfunc='size', fill_value=0)
        order_zone_matrix = zone_assign_df.pivot_table(index='ORD_NO', columns='ZONE', aggfunc='size', fill_value=0)
        order_zone_matrix = order_zone_matrix[sorted(order_zone_matrix.columns, key=lambda x: int(x.split('_')[-1]))]
       
        sku_sim = cosine_similarity(order_sku_matrix)
        zone_sim = cosine_similarity(order_zone_matrix)

        w_sku = 0.5
        w_zone = 0.5
        combined_sim = w_sku * sku_sim + w_zone * zone_sim        
        combined_dist = 1 - combined_sim # KMeans 거리 기반이기 때문
        
        n_orders = order_sku_matrix.shape[0]
        cart_capa = self.params.cart_capacity
        n_clusters = ceil(n_orders / cart_capa)
        
        kmeans = KMeans(n_clusters= n_clusters, random_state= 42)
        labels= kmeans.fit_predict(combined_dist)
        
        cluster_to_orders = defaultdict(list)
        for ord_no, label in zip(order_sku_matrix.index, labels) :
            cluster_to_orders[label].append(ord_no)
        
        all_orders= []
        for cluster_orders in cluster_to_orders.values() :
            all_orders.extend(cluster_orders)
                
        order_to_cart = {}
        cart_id = 1
        for i in range(0, len(all_orders), cart_capa) :
            batch = all_orders[i:i + cart_capa]
            for ord_no in batch :
                order_to_cart[ord_no] = cart_id
            cart_id += 1

        # for ord_no, cart_no in zip(order_sku_matrix.index, labels) :
        #     cart_no += 1
        #     order_to_cart[ord_no] = cart_no
 
        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

    def assign_batches_combination_sum_strategy(self) -> None:
        """
        (작업량 유사 & 카트 간 유사도 낮은)배치 단위로 피커-카트 배정
        """
        assigned_carts = set()
        batch_no = 1
        n_pickers = self.params.number_pickers
        carts_df = (
            self.orders.groupby('CART_NO')
            .agg(WORKLOAD=('SKU_CD', 'count'))
            .reset_index()
        )

        while len(assigned_carts) < carts_df.shape[0]:
            # 아직 배정되지 않은 cart 중 작업량이 가장 큰 cart 선택
            unassigned = carts_df[~carts_df['CART_NO'].isin(assigned_carts)]
            highest_workload_cart_row = unassigned.sort_values('WORKLOAD', ascending=False).iloc[0]
            highest_workload_cartno = highest_workload_cart_row['CART_NO']
            highest_workload_quantity = highest_workload_cart_row['WORKLOAD']

            batch = []
            pickers = {}

            # highest_workload_cart를 picker1에 할당
            pickers[1] = [highest_workload_cartno]
            batch.append(highest_workload_cartno)
            assigned_carts.add(highest_workload_cartno)

            # SKU 유사도 계산
            highest_workload_orders = self.orders[self.orders['CART_NO'] == highest_workload_cartno]
            highest_workload_skus = highest_workload_orders['SKU_CD'].unique()

            cart_similarities = []
            for _, row in unassigned.iterrows():
                if row['CART_NO'] == highest_workload_cartno:
                    continue
                candidate_orders = self.orders[self.orders['CART_NO'] == row['CART_NO']]
                candidate_skus = candidate_orders['SKU_CD'].unique()
                intersection = len(set(highest_workload_skus) & set(candidate_skus))
                union = len(set(highest_workload_skus) | set(candidate_skus))
                jaccard_sim = intersection / union if union > 0 else 0
                cart_similarities.append((row['CART_NO'], jaccard_sim, row['WORKLOAD']))

            # SKU 유사도가 먼 순으로 정렬 + 작업량이 높은 순으로 정렬
            cart_similarities.sort(key=lambda x: (x[1], -x[2]))

            # 나머지 피커에 배정할 cart 조합 찾기
            remaining_carts = [(c[0], c[2]) for c in cart_similarities]
            used_carts = set()

            for picker_id in range(2, n_pickers + 1):
                best_combo = None
                best_diff = float('inf')

                # 1~3개 조합 탐색
                for r in range(1, min(4, len(remaining_carts) + 1)):
                    for combo in combinations(remaining_carts, r):
                        combo_carts = [c[0] for c in combo]
                        combo_workload = sum(c[1] for c in combo)

                        if any(c in used_carts for c in combo_carts):
                            continue

                        diff = abs(combo_workload - highest_workload_quantity)
                        if diff < best_diff:
                            best_diff = diff
                            best_combo = combo_carts

                        if diff == 0:
                            break
                    if best_diff == 0:
                        break

                if best_combo:
                    pickers[picker_id] = best_combo
                    batch.extend(best_combo)
                    used_carts.update(best_combo)
                    assigned_carts.update(best_combo)

            # 더미 기록
            for picker, carts in pickers.items():
                for cart in carts:
                    self.orders.loc[self.orders['CART_NO'] == cart, 'PICKER_NO'] = picker
                    self.orders.loc[self.orders['CART_NO'] == cart, 'BATCH_NO'] = batch_no

            batch_no += 1

    def solve_picker_routing(self) -> None:
        """피커 경로 최적화
        1. 2-opt 초기해 설정
        2. Simulated Annealing 근사해 최적화"""
        
        # 경로 전체 합
        def calculate_total_distance(route, dist_matrix):
            return sum(dist_matrix.loc[route[i], route[i + 1]] for i in range(len(route) - 1))

        # 경로 뒤집어 보기
        def two_opt(route, dist_matrix):
            best = route
            improved = True
            while improved:
                improved = False
                for i in range(1, len(best) - 2): # start 지점 제외
                    for j in range(i + 1, len(best) - 1):  # end 지점 제외
                        # 1인 경우, 뒤집을 부분이 없으므로 다음 for문 진행
                        if j - i == 1:
                            continue
                        # 경로 뒤집어보기
                        new_route = best[:i] + best[i:j][::-1] + best[j:]
                        if calculate_total_distance(new_route, dist_matrix) < calculate_total_distance(best, dist_matrix):
                            best = new_route
                            improved = True
                if improved:
                    break
            return best

        def simulated_annealing(route, dist_matrix, initial_temp=1000, cooling_rate=0.995, min_temp=1):
            current_route = route
            current_distance = calculate_total_distance(current_route, dist_matrix)
            best_route = list(current_route)
            best_distance = current_distance
            temperature = initial_temp

            while temperature > min_temp:
                i, j = sorted(np.random.choice(range(1, len(route) - 1), 2, replace=False))  # exclude start/end
                new_route = current_route[:i] + current_route[i:j][::-1] + current_route[j:]
                new_distance = calculate_total_distance(new_route, dist_matrix)

                # 경로가 좋아졌거나 새 경로가 좋지 않아도 일정 확률로 받아들임
                # 점점 온도가 올라가면서 받아들일 확률이 높아짐
                if new_distance < current_distance or np.random.random() < np.exp((current_distance - new_distance) / temperature):
                    current_route = new_route
                    current_distance = new_distance
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance

                temperature *= cooling_rate
            return best_route

        # 실제 적용: 카트 단위로 LOC 최적화
        self.orders['SEQ'] = 0
        cart_groups = self.orders.groupby('CART_NO')
        for cart_no, cart_df in cart_groups:
            locs = cart_df['LOC'].dropna().unique().tolist()
            if len(locs) <= 1:
                route = locs
            else:
                # 전체 경로: START + LOCs + END
                route = [self.start_location] + locs + [self.end_location]
                route = two_opt(route, self.od_matrix)
                route = simulated_annealing(route, self.od_matrix)
                route = [loc for loc in route if loc not in [self.start_location, self.end_location]]  # SEQ는 중간 LOC만

            seq = 1
            for loc in route:
                mask = (self.orders['CART_NO'] == cart_no) & (self.orders['LOC'] == loc)
                loc_indices = self.orders[mask].index.tolist()
                for idx in loc_indices:
                    self.orders.at[idx, 'SEQ'] = seq
                    seq += 1

    def solve(self) -> pd.DataFrame:
        """Execute complete warehouse optimization solution"""
        self.solve_storage_location()
        self.solve_order_batching()
        self.assign_batches_combination_sum_strategy()
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
    return solver.solve()

if __name__ == "__main__":
    import time
    test_INPUT = pd.read_csv("Sample_InputData.csv")
    test_PARAM = pd.read_csv("sample_Parameters.csv")
    test_OD = pd.read_csv("sample_OD_Matrix.csv", index_col=0, header=0)
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