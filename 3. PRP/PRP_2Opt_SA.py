import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations
from statistics import mode
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from math import ceil
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import re
from joblib import Parallel, delayed
import heapq
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
        self.sku_sim = None
        self.zone_sim = None
        self.order_labels = None
        self._initialize_orders()
        self._validate_input()
        self.picker_routes = defaultdict(list)  # picker별 경로 저장

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
        """Solve Storage Location Assignment Problem (SLAP) using SKU frequency and co-ordered
        1. 상위 20% SKU를 출고빈도 기준으로 정렬
        2. 빈도순으로 차례로 꺼내고
        - 자신이 입고지에서 가장 가까운 랙을 먼저 선점
        - 공동 주문된 SKU들을 현재 랙 근처 가장 가까운 랙으로 Greedy 재배치

        남은 SKU 중애 공동 주문된 SKU가 이미 배치되어 있다면
        해당 랙들 근처(이미 배치된 sku가 많다면(<=> 랙이 많다면) 평균 거리)로 그리디 배치 
        남은 랙, 남은 SKU에 대해서는 빈도수 기반으로 입출고 지점과 가깝게 배치"""
        
        def assgin_sku_to_loc(top_percent = 0.2) :        
            sku_freq = self.orders['SKU_CD'].value_counts()
            top_k =int(len(sku_freq) * top_percent)
            top_skus = sku_freq.head(top_k).index.tolist()

            # {SKU : {ORD, ,,}}
            sku_orders = self.orders.groupby('SKU_CD')['ORD_NO'].apply(set).to_dict()
            # sku_orders

            cooc_sku_map = defaultdict(set)
            for a in top_skus :
                for b in sku_orders :
                    if a == b :
                        continue
                    # 등장했던 주문 중 몇 번이나 함께 등장했는지
                    inter = sku_orders[a] & sku_orders[b]
                    # a와 b가 등장했던 총 주문
                    union = sku_orders[a] | sku_orders[b]
                    if union and len(inter) > 0 :
                        cooc_sku_map[a].add(b)

            # cooc_sku_map - {빈도 높은 SKU : {함께 등장한 적 있는 SKU}}
            racks = self.od_matrix.index[2:]
            start_to_rack_dists=  self.od_matrix.loc[self.od_matrix.index[0], racks]
            rack_sorted = start_to_rack_dists.sort_values().index.tolist()

            assigned_skus = set()
            # {rack : count}
            rack_assign_count = {rack: 0 for rack in rack_sorted}
            sku_to_loc = {}

            # step1 : 상위 20% SKU 입출고 지점 + 공동 주문 SKU 가까이 배치
            # 상위 20% SKU 입출고 지점 우선 배치
            for sku in top_skus :
                if sku in assigned_skus :
                    continue
                for rack in rack_sorted :
                    if rack_assign_count[rack] < self.params.rack_capacity :
                        sku_to_loc[sku] = rack
                        assigned_skus.add(sku)
                        rack_assign_count[rack] += 1
                        base_rack = rack
                        break
              
                # 공동 주문된 SKU들 가까이 배치
                neighbors = cooc_sku_map[sku]
                if not neighbors :
                    continue
            
                # 기준 랙과의 거리
                base_rack_dist = self.od_matrix.loc[base_rack, racks]
                near_racks_sorted = base_rack_dist.sort_values().index.tolist()

                for nb_sku in neighbors :
                    if nb_sku in assigned_skus :
                      continue
                    for rack in near_racks_sorted :
                        if rack_assign_count[rack] < self.params.rack_capacity :
                            sku_to_loc[nb_sku] = rack
                            assigned_skus.add(nb_sku)
                            rack_assign_count[rack] += 1
                            break
                
            # Step2 : 이미 배치되어 있는 sku와 공동 주문된 sku가 있다면 근처 배치
            remaining_skus = [s for s in sku_freq.index if s not in assigned_skus]
            for remained_sku in remaining_skus :
                # {ORD, ORD, ,,} - remaining_sku별 등장한 주문 목록
                remained_sku_orders = sku_orders.get(remained_sku, set())
                co_ordered_assigned_locs = []
                for assigned_sku in sku_to_loc :
                    if assigned_sku == remained_sku :
                        continue
                    # 공동 주문된 ORD가 있다면
                    if remained_sku_orders & sku_orders[assigned_sku] :
                        co_ordered_assigned_locs.append(sku_to_loc[assigned_sku])

                # remaining_sku가 이미 배치된 SKU와 공동 주문된 SKU가 있다면
                # {RACK, RACK, ,,}
                if co_ordered_assigned_locs :
                    # 공동 주문된 SKU의 LOC까지의 평균 거리
                    # RACK_1 : 33, ,, ~ RACK_168 : 26
                    avg_dist = self.od_matrix.loc[racks, co_ordered_assigned_locs].mean(axis= 1)
                    candidate_racks = avg_dist.sort_values().index.tolist()
                # remaining_sku가 공동 주문된 SKU가 없다면
                else :
                    candidate_racks = rack_sorted

                for rack in candidate_racks :
                    if rack_assign_count[rack] < self.params.rack_capacity :
                        assigned_skus.add(remained_sku)
                        sku_to_loc[remained_sku] = rack
                        rack_assign_count[rack] += 1
                        break
            
            final_remained_skus = [s for s in sku_freq.index if s not in sku_to_loc]
            print('할당되지 않은 SKU:', len(final_remained_skus))
            final_remained_racks = [rack for rack, count in rack_assign_count.items() if count < self.params.rack_capacity]
            print('빈자리가 남은 랙:', len(final_remained_racks))
            
            return sku_to_loc
        
        sku_to_loc = assgin_sku_to_loc(top_percent= 0.2)
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_loc)

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

        order_sku_matrix = zone_assign_df.pivot_table(index='ORD_NO', columns='SKU_CD', aggfunc='size', fill_value=0)
        order_zone_matrix = zone_assign_df.pivot_table(index='ORD_NO', columns='ZONE', aggfunc='size', fill_value=0)
        order_zone_matrix = order_zone_matrix[sorted(order_zone_matrix.columns, key=lambda x: int(x.split('_')[-1]))]

        sku_sim = cosine_similarity(order_sku_matrix)
        zone_sim = cosine_similarity(order_zone_matrix)
        self.sku_sim = sku_sim
        self.zone_sim = zone_sim
        combined_sim = 0.1 * sku_sim + 0.9 * zone_sim
        combined_dist = 1 - combined_sim

        n_orders = order_sku_matrix.shape[0]
        cart_capa = self.params.cart_capacity
        n_clusters = ceil(n_orders / cart_capa)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(combined_dist)

        cluster_to_orders = defaultdict(list)
        for ord_no, label in zip(order_sku_matrix.index, labels):
            cluster_to_orders[label].append(ord_no)

        all_orders = []
        for cluster_orders in cluster_to_orders.values():
            all_orders.extend(cluster_orders)

        order_to_cart = {}
        cart_id = 1
        for i in range(0, len(all_orders), cart_capa):
            batch = all_orders[i:i + cart_capa]
            for ord_no in batch:
                order_to_cart[ord_no] = cart_id
            cart_id += 1

        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)


    def assign_carts_balanced(self) -> pd.DataFrame:
        """
        피커들이 작업을 비슷한 시간에 끝내도록 작업량을 균형 있게 배분하며,
        피커들이 한 번씩 할당받을 때마다 새로운 배치 번호를 부여합니다.

        Returns:
        - self.orders: PICKER_NO, BATCH_NO 칼럼이 추가된 DataFrame
        """

        # CART별 예상 작업시간 계산
        cart_times = {}
        for cart_no, group in self.orders.groupby('CART_NO'):
            locs = group['LOC'].unique()
            travel_dist = sum(
                self.od_matrix.loc[self.od_matrix.index[0], loc] for loc in locs
            )
            picking_time = len(group) * self.params.picking_time
            walking_time = travel_dist * self.params.walking_time
            cart_times[cart_no] = picking_time + walking_time

        # CART 작업량 기준으로 내림차순 정렬
        remaining_carts = sorted(cart_times.items(), key=lambda x: -x[1])

        # 피커 상태: (총 작업시간, 피커번호)
        picker_heap = [(0, picker_no) for picker_no in range(1, self.params.number_pickers + 1)]
        heapq.heapify(picker_heap)

        batch_no = 1
        cart_batch_map = {}
        cart_picker_map = {}
        assigned_in_current_batch = 0

        while remaining_carts:
            cart_no, work_time = remaining_carts.pop(0)

            # 현재 작업량이 가장 적은 피커에게 할당
            current_time, picker_no = heapq.heappop(picker_heap)

            cart_batch_map[cart_no] = batch_no
            cart_picker_map[cart_no] = picker_no

            # 피커 작업량 갱신 후 다시 힙에 넣기
            heapq.heappush(picker_heap, (current_time + work_time, picker_no))

            assigned_in_current_batch += 1

            # n_pickers개가 모두 할당되면 batch_no 증가
            if assigned_in_current_batch == self.params.number_pickers:
                batch_no += 1
                assigned_in_current_batch = 0

        # 결과 기록
        for cart_no, picker_no in cart_picker_map.items():
            self.orders.loc[self.orders['CART_NO'] == cart_no, 'PICKER_NO'] = picker_no
            self.orders.loc[self.orders['CART_NO'] == cart_no, 'BATCH_NO'] = cart_batch_map[cart_no]

        return self.orders



    def remapping_cart_no(self):
        # ✅ 루프 종료 후 CART_NO 재매핑
        self.orders = self.orders.sort_values(['BATCH_NO', 'PICKER_NO'])

        unique_cart_combinations = (
            self.orders
            .drop_duplicates(subset=['BATCH_NO', 'PICKER_NO', 'CART_NO'])
            .sort_values(['BATCH_NO', 'PICKER_NO'])
        )

        cart_mapping_ordered = {
            row.CART_NO: new_cart_no
            for new_cart_no, row in enumerate(unique_cart_combinations.itertuples(index=False), start=1)
        }

        self.orders['CART_NO'] = self.orders['CART_NO'].map(cart_mapping_ordered)




    def solve_picker_routing_parallel(self) -> None:
        """피커 경로 최적화 (병렬처리)
        - 2-opt + Simulated Annealing
        """

        def calculate_total_distance(route, dist_matrix):
            return sum(dist_matrix.loc[route[i], route[i + 1]] for i in range(len(route) - 1))

        def two_opt(route, dist_matrix):
            best = route
            improved = True
            while improved:
                improved = False
                for i in range(1, len(best) - 2):
                    for j in range(i + 1, len(best) - 1):
                        if j - i == 1:
                            continue
                        new_route = best[:i] + best[i:j][::-1] + best[j:]
                        if calculate_total_distance(new_route, dist_matrix) < calculate_total_distance(best, dist_matrix):
                            best = new_route
                            improved = True
                if improved:
                    break
            return best

        def simulated_annealing(route, dist_matrix, initial_temp=1000, cooling_rate=0.995, min_temp=0.01):
            current_route = route
            current_distance = calculate_total_distance(current_route, dist_matrix)
            best_route = list(current_route)
            best_distance = current_distance
            temperature = initial_temp

            while temperature > min_temp:
                i, j = sorted(np.random.choice(range(1, len(route) - 1), 2, replace=False))
                new_route = current_route[:i] + current_route[i:j][::-1] + current_route[j:]
                new_distance = calculate_total_distance(new_route, dist_matrix)

                if new_distance < current_distance or np.random.random() < np.exp((current_distance - new_distance) / temperature):
                    current_route = new_route
                    current_distance = new_distance
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance

                temperature *= cooling_rate

            return best_route

# 30회 반복
#  ✅ 총 피킹 시간: 19566.57초 (피킹 1426회, 이동 거리 15288.57)

        def optimize_cart_route(cart_no, cart_df, od_matrix, n_trials= 15):
            locs = cart_df['LOC'].dropna().unique().tolist()
            if len(locs) <= 1:
                route = locs
            else:
                best_route = None
                best_distance = float('inf')
                
                for _ in range(n_trials) :
                    route_trial = [od_matrix.index[0]] + locs + [od_matrix.index[1]]
                    np.random.shuffle(route_trial[1:-1]) # loc 순서 무작위화
                    route_trial = two_opt(route_trial, od_matrix)
                    route_trial = simulated_annealing(route_trial, od_matrix)
                    
                    distance= calculate_total_distance(route_trial, od_matrix)
                    if distance < best_distance :
                        best_distance = distance
                        best_route = route_trial
                
                route = [loc for loc in best_route if loc not in [od_matrix.index[0], od_matrix.index[1]]]

            seq_records = []
            seq = 1
            for loc in route:
                mask = (cart_df['LOC'] == loc)
                loc_indices = cart_df[mask].index.tolist()
                for idx in loc_indices:
                    seq_records.append((idx, seq))
                    seq += 1
            return seq_records


        cart_groups = list(self.orders.groupby('CART_NO'))

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(optimize_cart_route)(cart_no, cart_df, self.od_matrix) for cart_no, cart_df in cart_groups
        )

        # 결과 반영
        all_seq_records = [record for sublist in results for record in sublist]
        seq_df = pd.DataFrame(all_seq_records, columns=['index', 'SEQ']).set_index('index')
        self.orders.loc[seq_df.index, 'SEQ'] = seq_df['SEQ']

    def calculate_total_picking_time(self) -> float:
        total_walking_distance = 0.0
        total_picking_count = len(self.orders)

        for cart_no, group in self.orders.groupby('CART_NO'):
            group_sorted = group.sort_values('SEQ')
            locations = [self.start_location] + [self.end_location] + group_sorted['LOC'].tolist()

            for i in range(len(locations) - 1):
                from_loc, to_loc = locations[i], locations[i + 1]

                # 🔍 존재 여부 확인
                if from_loc not in self.od_matrix.index or to_loc not in self.od_matrix.columns:
                    print(f"❌ 경로 조회 오류: from={from_loc}, to={to_loc}")
                    raise ValueError("OD Matrix에 존재하지 않는 위치입니다.")

                dist = self.od_matrix.loc[from_loc, to_loc]

                # 🔍 거리 값이 NaN이면 경고
                if pd.isna(dist):
                    print(f"❗ NaN 거리 발생: from={from_loc}, to={to_loc}")
                    raise ValueError("OD Matrix에서 NaN 거리값 발견")

                total_walking_distance += dist

        total_time = (
            total_walking_distance * self.params.walking_time +
            total_picking_count * self.params.picking_time
        )
        print(f"\n ✅ 총 피킹 시간: {total_time:.2f}초 "
              f"(피킹 {total_picking_count}회, 이동 거리 {total_walking_distance:.2f})")

    def solve(self) -> pd.DataFrame:
        self.solve_storage_location()
        self.solve_order_batching()
        self.assign_carts_balanced()
        self.remapping_cart_no()
        self.solve_picker_routing_parallel()
        self.calculate_total_picking_time()
        return self.orders
def main(INPUT: pd.DataFrame, PARAMETER: pd.DataFrame, OD_MATRIX: pd.DataFrame) -> pd.DataFrame:
    solver = WarehouseSolver(INPUT, PARAMETER, OD_MATRIX)
    result = solver.solve()
    return result

if __name__ == "__main__":
    test_INPUT = pd.read_csv("./data/Sample_InputData.csv")
    test_PARAM = pd.read_csv("./data/Sample_Parameters.csv")
    test_OD = pd.read_csv("./data/Sample_OD_Matrix.csv", index_col=0, header=0)
    result = main(test_INPUT, test_PARAM, test_OD)
    result.to_csv("OutputData.csv", index=False)
