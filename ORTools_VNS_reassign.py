import pandas as pd
import numpy as np
import time
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
    def SLAP(self) -> None:
        """Solve Storage Location Assignment Problem (SLAP) using SKU frequency and co-ordered
        1. 상위 20% SKU를 출고빈도 기준으로 정렬
        2. 빈도순으로 차례로 꺼내고
        - 자신이 입고지에서 가장 가까운 랙을 먼저 선점
        - 공동 주문된 SKU들을 현재 랙 근처 가장 가까운 랙으로 Greedy 재배치

        남은 SKU 중애 공동 주문된 SKU가 이미 배치되어 있다면
        해당 랙들 근처(이미 배치된 sku가 많다면(<=> 랙이 많다면) 평균 거리)로 그리디 배치 
        남은 랙, 남은 SKU에 대해서는 빈도수 기반으로 입출고 지점과 가깝게 배치"""
        
        def assgin_sku_to_loc_by_2step(top_percent = 0.2) :        
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
        
        sku_to_loc = assgin_sku_to_loc_by_2step(top_percent= 0.2)
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_loc)
      

        # ✅ ZONE 할당
        def extract_loc_num(loc_name):
            import re
            match = re.search(r'\d+', str(loc_name))
            return int(match.group()) if match else None

        def estimate_row_count_from_od_matrix(od_matrix):
            """
            OD Matrix에서 한 존의 랙 개수를 추정
            """
            row_len_candidates = []
            for rack in od_matrix.index[2:]:
                distances = od_matrix.loc[rack].sort_values()
                diffs = distances.diff().fillna(0)
                jump_indices = (diffs > 5).to_numpy().nonzero()[0]
                if len(jump_indices) > 0:
                    row_len_candidates.append(jump_indices[0])
            from statistics import mode
            est_rack_count = mode(row_len_candidates) if row_len_candidates else 10
            return est_rack_count

        # LOC_NUM 컬럼 생성
        self.orders['LOC_NUM'] = self.orders['LOC'].map(extract_loc_num)

        # 존 크기 추정
        rack_count_per_zone = estimate_row_count_from_od_matrix(self.od_matrix)

        # ZONE 할당
        self.orders['ZONE'] = self.orders['LOC_NUM'].apply(
            lambda x: f"ZONE_{(x - 1) // rack_count_per_zone + 1}" if pd.notnull(x) else None
        )
    ##### 2) OBSP
    ##### 2-1) OBSP(ORD->CART)
    def OBSP_ORD_to_CART(self) -> None:
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
        combined_sim = 0.5 * sku_sim + 0.5 * zone_sim
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

    ##### 2-2)OBSP(CART->BATCH + CART SEQ)
    # ORTools + VNS
    # 카트 내 주문 픽업 순서(경로 최소화)
    def OBSP_CART_to_BATCH(self) -> None:
        """
        PRP + 작업량 균형 배분:
        각 카트별로 경로를 OR-Tools로 초기화하고 VNS로 개선한 후,
        작업량을 균형 있게 피커들에게 배분하는 하이브리드 메서드.
        """
        import os
        from joblib import Parallel, delayed
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        def calculate_cart_workload(cart_df, route) -> float:
            depot = self.start_location
            travel = sum(
                self.od_matrix.loc[a, b]
                for a, b in zip([depot] + route, route + [depot])
            ) * self.params.walking_time
            picking = len(cart_df) * self.params.picking_time
            return travel + picking

        def vns(batch_df: pd.DataFrame, initial_order: list, max_k=3, max_iter=50, patience=5) -> list:
            """
            Variable Neighborhood Search (VNS) with 3 neighborhoods and optional early stopping.

            Args:
                batch_df: 작업 데이터프레임
                initial_order: 초기 카트 순서
                max_k: 최대 neighborhood 크기 (1=swap, 2=reverse, 3=relocate)
                max_iter: 최대 반복 횟수
                patience: 개선이 없는 반복 허용 횟수 (early stopping)

            Returns:
                best_order: 최적의 카트 순서
            """
            best_order = initial_order[:]
            best_score = calculate_cart_workload(batch_df, best_order)

            no_improve_count = 0

            for _ in range(max_iter):
                if no_improve_count >= patience:
                    break

                k = 1
                improved_in_iter = False

                while k <= max_k:
                    new_order = best_order[:]
                    i, j = sorted(np.random.choice(len(new_order), 2, replace=False))

                    if k == 1:
                        # Swap two carts
                        new_order[i], new_order[j] = new_order[j], new_order[i]
                    elif k == 2:
                        # Reverse a subsequence
                        new_order = new_order[:i] + new_order[i:j+1][::-1] + new_order[j+1:]
                    elif k == 3:
                        # Relocate one cart to another position
                        node = new_order.pop(i)
                        new_order.insert(j, node)

                    new_score = calculate_cart_workload(batch_df, new_order)

                    if new_score < best_score:
                        best_order = new_order
                        best_score = new_score
                        k = 1  # reset neighborhood
                        improved_in_iter = True
                    else:
                        k += 1

                if improved_in_iter:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            return best_order

        def optimize_cart(cart_no: int, cart_df: pd.DataFrame):
            locs = cart_df['LOC'].dropna().unique().tolist()
            depot = self.start_location

            if len(locs) <= 1:
                route = locs
            else:
                full_locs = [depot] + locs + [depot]

                # OR-Tools 초기화
                manager = pywrapcp.RoutingIndexManager(len(full_locs), 1, 0)
                routing = pywrapcp.RoutingModel(manager)

                dist_matrix = np.zeros((len(full_locs), len(full_locs)))
                for i, from_loc in enumerate(full_locs):
                    for j, to_loc in enumerate(full_locs):
                        dist_matrix[i, j] = self.od_matrix.loc[from_loc, to_loc]

                def distance_callback(from_index, to_index):
                    return int(dist_matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)] * 1000)

                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
                search_parameters.time_limit.seconds = 1

                solution = routing.SolveWithParameters(search_parameters)

                if solution:
                    ortools_route = []
                    index = routing.Start(0)
                    while not routing.IsEnd(index):
                        node = manager.IndexToNode(index)
                        if node != 0 and node != len(full_locs) - 1:
                            ortools_route.append(locs[node-1])
                        index = solution.Value(routing.NextVar(index))
                else:
                    ortools_route = locs

                route = vns(cart_df, ortools_route)

            workload = calculate_cart_workload(cart_df, route)
            ords = cart_df['ORD_NO'].unique()
            seq_records = []
            seq = 1
            for loc in route:
                for idx in cart_df[cart_df['LOC'] == loc].index.tolist():
                    seq_records.append((idx, seq))
                    seq += 1

            return cart_no, workload, set(ords), seq_records

        cart_groups = list(self.orders.groupby('CART_NO'))
        n_jobs = min(len(cart_groups), os.cpu_count())

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(optimize_cart)(cart_no, cart_df) for cart_no, cart_df in cart_groups
        )

        cart_times, cart_orders, cart_nos, seq_all = {}, {}, [], []
        for cart_no, workload, ords, seq_records in results:
            cart_times[cart_no] = workload
            cart_orders[cart_no] = ords
            cart_nos.append(cart_no)
            seq_all.extend(seq_records)

        seq_df = pd.DataFrame(seq_all, columns=['index', 'SEQ']).set_index('index')
        self.orders.loc[seq_df.index, 'SEQ'] = seq_df['SEQ']

        # 🧩 작업량 균형 배분
        cart_picker_map, cart_batch_map = self.OBSP_BATCH_BALANCED(
            cart_nos=cart_nos,
            cart_times=cart_times,
            cart_orders=cart_orders,
            num_pickers=self.params.number_pickers
        )

        for cart_no, picker_no in cart_picker_map.items():
            self.orders.loc[self.orders['CART_NO'] == cart_no, 'PICKER_NO'] = picker_no
            self.orders.loc[self.orders['CART_NO'] == cart_no, 'BATCH_NO'] = cart_batch_map[cart_no]

    ##### 2-2-1)OBSP(Balanced BATCH)
    # 경로 최소화된 결과를 바탕으로 BATCH내 PICKER 작업 균형
    def OBSP_BATCH_BALANCED(self, cart_nos, cart_times, cart_orders, num_pickers):
        """
        카트를 피커에게 균형 있게 배분하면서 유사도가 낮은 카트끼리 묶는 함수.

        Args:
            cart_nos (list): 카트 번호 리스트
            cart_times (dict): 카트별 작업량 {cart_no: workload}
            cart_orders (dict): 카트별 주문번호 집합 {cart_no: set(orders)}
            num_pickers (int): 피커 수

        Returns:
            tuple: (cart_picker_map, cart_batch_map)
                cart_picker_map (dict): {cart_no: picker_no}
                cart_batch_map (dict): {cart_no: batch_no}
        """
        # 카트 유사도 계산
        n_carts = len(cart_nos)
        cart_sim = np.zeros((n_carts, n_carts))
        for i, ci in enumerate(cart_nos):
            for j, cj in enumerate(cart_nos):
                if i >= j: continue
                oi, oj = cart_orders[ci], cart_orders[cj]
                inter, union = len(oi & oj), len(oi | oj)
                sim = inter / union if union else 0.0
                cart_sim[i, j] = cart_sim[j, i] = sim

        # 피커 초기 할당
        P, best_combo, remaining = num_pickers, [], set(range(n_carts))
        first_idx = max(remaining, key=lambda idx: cart_times[cart_nos[idx]])
        best_combo.append(first_idx)
        remaining.remove(first_idx)

        while len(best_combo) < P:
            next_idx = min(remaining, key=lambda idx: sum(cart_sim[idx, chosen] for chosen in best_combo))
            best_combo.append(next_idx)
            remaining.remove(next_idx)

        # 균형 배분
        assigned_carts, picker_heap, cart_batch_map, cart_picker_map, batch_no = set(), [], {}, {}, 1

        for picker_no, idx in enumerate(best_combo, start=1):
            cart_no = cart_nos[idx]
            assigned_carts.add(cart_no)
            heapq.heappush(picker_heap, (cart_times[cart_no], picker_no, cart_no))
            cart_batch_map[cart_no] = batch_no
            cart_picker_map[cart_no] = picker_no

        remaining_carts = [c for c in cart_nos if c not in assigned_carts]

        while remaining_carts:
            current_time, picker_no, last_cart_no = heapq.heappop(picker_heap)
            last_cart_idx = cart_nos.index(last_cart_no)

            best_next_cart, lowest_sim, best_workload_diff = None, float('inf'), float('inf')

            for c in remaining_carts:
                idx = cart_nos.index(c)
                sim = cart_sim[last_cart_idx, idx]
                projected_time = current_time + cart_times[c]
                if sim < lowest_sim or (sim == lowest_sim and projected_time < best_workload_diff):
                    lowest_sim, best_workload_diff, best_next_cart = sim, projected_time, c

            assigned_carts.add(best_next_cart)
            remaining_carts.remove(best_next_cart)

            new_time = current_time + cart_times[best_next_cart]
            heapq.heappush(picker_heap, (new_time, picker_no, best_next_cart))
            cart_batch_map[best_next_cart] = batch_no
            cart_picker_map[best_next_cart] = picker_no

            if len(assigned_carts) % P == 0:
                batch_no += 1

        return cart_picker_map, cart_batch_map
    
    ##### 3) PRP
    # ORTools + VNS
    # Batch내에 카트 순서(Blocking_time 최소화)
    def PRP(self) -> None:
            """
            PRP 단계: 배치별로 카트 순서를 OR-Tools로 초기화하고,
            VNS를 후처리로 적용해 blocking time을 최소화 (하이브리드)
            """
            import os
            from joblib import Parallel, delayed
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
            from collections import defaultdict

            def calculate_blocking_time(batch_df: pd.DataFrame, cart_order: list) -> float:
                timeline = defaultdict(list)
                for cart_no in cart_order:
                    cart = batch_df[batch_df['CART_NO'] == cart_no].sort_values('SEQ')
                    time = 0.0
                    last_loc = self.start_location
                    for _, row in cart.iterrows():
                        loc = row['LOC']
                        time += self.od_matrix.loc[last_loc, loc] * self.params.walking_time
                        start_time = time
                        time += self.params.picking_time
                        end_time = time
                        timeline[loc].append((start_time, end_time))
                        last_loc = loc
                    time += self.od_matrix.loc[last_loc, self.end_location] * self.params.walking_time

                blocking_time = 0.0
                for loc, intervals in timeline.items():
                    intervals.sort()
                    active = []
                    for start, end in intervals:
                        active = [e for e in active if e > start]
                        blocking_time += len(active) * (end - start)
                        active.append(end)
                return blocking_time

            def vns(batch_df: pd.DataFrame, initial_order: list, max_k=3, max_iter=50, patience=5) -> list:
                """
                Variable Neighborhood Search (VNS) with 3 neighborhoods and optional early stopping.

                Args:
                    batch_df: 작업 데이터프레임
                    initial_order: 초기 카트 순서
                    max_k: 최대 neighborhood 크기 (1=swap, 2=reverse, 3=relocate)
                    max_iter: 최대 반복 횟수
                    patience: 개선이 없는 반복 허용 횟수 (early stopping)

                Returns:
                    best_order: 최적의 카트 순서
                """
                best_order = initial_order[:]
                best_score = calculate_blocking_time(batch_df, best_order)

                no_improve_count = 0

                for _ in range(max_iter):
                    if no_improve_count >= patience:
                        break

                    k = 1
                    improved_in_iter = False

                    while k <= max_k:
                        new_order = best_order[:]
                        i, j = sorted(np.random.choice(len(new_order), 2, replace=False))

                        if k == 1:
                            # Swap two carts
                            new_order[i], new_order[j] = new_order[j], new_order[i]
                        elif k == 2:
                            # Reverse a subsequence
                            new_order = new_order[:i] + new_order[i:j+1][::-1] + new_order[j+1:]
                        elif k == 3:
                            # Relocate one cart to another position
                            node = new_order.pop(i)
                            new_order.insert(j, node)

                        new_score = calculate_blocking_time(batch_df, new_order)

                        if new_score < best_score:
                            best_order = new_order
                            best_score = new_score
                            k = 1  # reset neighborhood
                            improved_in_iter = True
                        else:
                            k += 1

                    if improved_in_iter:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                return best_order

            def optimize_batch(batch_no: int, batch_df: pd.DataFrame) -> pd.DataFrame:
                cart_nos = batch_df['CART_NO'].unique().tolist()
                depot_idx = 0

                # 카트 간 거리 근사 행렬
                dist_matrix = np.zeros((len(cart_nos), len(cart_nos)))
                for i, cart_i in enumerate(cart_nos):
                    for j, cart_j in enumerate(cart_nos):
                        if i == j:
                            dist_matrix[i, j] = 0
                        else:
                            loc_seq_i = batch_df[batch_df['CART_NO'] == cart_i]['SEQ'].tolist()
                            loc_seq_j = batch_df[batch_df['CART_NO'] == cart_j]['SEQ'].tolist()
                            dist_matrix[i, j] = abs(min(loc_seq_j) - min(loc_seq_i))

                # OR-Tools TSP
                manager = pywrapcp.RoutingIndexManager(
                    len(cart_nos), 1, depot_idx
                )
                routing = pywrapcp.RoutingModel(manager)

                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return int(dist_matrix[from_node][to_node] * 1000)

                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
                search_parameters.time_limit.seconds = 1

                solution = routing.SolveWithParameters(search_parameters)

                if solution:
                    ortools_order = []
                    index = routing.Start(0)
                    while not routing.IsEnd(index):
                        node = manager.IndexToNode(index)
                        ortools_order.append(cart_nos[node])
                        index = solution.Value(routing.NextVar(index))
                else:
                    ortools_order = cart_nos  # fallback

                # VNS 후처리
                final_order = vns(batch_df, ortools_order)

                optimized_df = pd.concat(
                    [batch_df[batch_df['CART_NO'] == cart_no] for cart_no in final_order]
                )
                return optimized_df

            batch_groups = list(self.orders.groupby('BATCH_NO'))
            n_jobs = min(len(batch_groups), os.cpu_count())

            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(optimize_batch)(batch_no, batch_df) for batch_no, batch_df in batch_groups
            )

            self.orders = pd.concat(results).sort_index()

    ### etc) CART_NO Remapping
    def REMAPPING_CART(self):
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

    # def reassign_sku_by_cart(self) :
    #     od_mat = self.od_matrix
    #     orders = self.orders.copy()
    #     start_location = self.start_location
    #     rack_capa = self.params.rack_capacity
    #     racks = od_mat.index[2:]
        
    #     sku_freq = orders['SKU_CD'].value_counts() # sku_cd 10
    #     sku_freq.head()

    #     cart_skus = orders.groupby('CART_NO')['SKU_CD'].apply(list).to_dict()
    #     cart_skus # cart_no : [sku_list]

    #     sku_seq = orders.groupby('SKU_CD')['SEQ'].min().to_dict()
    #     sku_seq # sku_cd : min(seq)

    #     assigned_sku = set()
    #     rack_assign_count = {rack: 0 for rack in racks}
    #     new_sku_to_loc = {}

    #     # 대표 SKU 후보
    #     cart_main_sku = {}
    #     for cart, sku_list in cart_skus.items() :
    #       sku_list = list(set(sku_list))
    #       # 대표 SKU : 빈도 높은순 / 앞 순서 순
    #       sku_list.sort(key= lambda x : (-sku_freq.get(x, 0), sku_seq.get(x, float('inf'))))
    #       cart_main_sku[cart] = sku_list[0]

    #     cart_main_sku # 'CART_NO : main_SKU
    #     # 대표 SKU 우선순위(-빈도, 앞 순서)로 CART 정렬
    #     sorted_cart_by_main_sku = sorted(cart_main_sku.items(),
    #                                      key= lambda x : (-sku_freq.get(x[1], 0), sku_seq.get(x[1], float('inf'))))
    #     sorted_cart_by_main_sku # (빈도 높고 앞 순서인 대표 SKU를 포함한 카트)

    #     for cart, main_sku in sorted_cart_by_main_sku :
    #       cart_df = orders[orders['CART_NO'] == cart].copy()
    #       cart_df = cart_df.sort_values('SEQ')
    #       cart_skus = cart_df['SKU_CD'].unique().tolist()

    #       if main_sku not in assigned_sku :
    #         for rack in od_mat.loc[start_location, racks].sort_values().index :
    #           if rack_assign_count[rack] < rack_capa :
    #             new_sku_to_loc[main_sku] = rack 
    #             assigned_sku.add(main_sku)
    #             rack_assign_count[rack] += 1 
    #             base_rack = rack 
    #             break 
    #       else :
    #         base_rack = new_sku_to_loc[main_sku]

    #       remaining_skus = [sku for sku in cart_skus if sku!= main_sku and sku not in assigned_sku]
    #       # 빈도순 / 앞 순서순
    #       remaining_skus.sort(key= lambda x: (-sku_freq.get(x, 0), sku_seq.get(x, float('inf'))))

    #       for sku in remaining_skus :
    #         nearby_racks = od_mat.loc[base_rack, racks].sort_values().index
    #         for rack in nearby_racks :
    #           if rack_assign_count[rack] < rack_capa :
    #             new_sku_to_loc[sku] = rack 
    #             assigned_sku.add(sku)
    #             rack_assign_count[rack] += 1 
    #             break 
            
    #     self.orders['LOC'] = self.orders['SKU_CD'].map(new_sku_to_loc)
    #     print("✅ 미배치 SKU 수:", len(set(orders['SKU_CD']) - set(new_sku_to_loc)))
    #     print("✅ 남은 빈 랙 수:", sum(v < rack_capa for v in rack_assign_count.values()))

    def reassign_sku_by_cart_similarity_clustered(self):
        from sklearn.metrics import pairwise_distances
        from sklearn.cluster import KMeans

        orders = self.orders.copy()
        racks = self.od_matrix.index[2:]
        rack_capa = self.params.rack_capacity
        od_mat = self.od_matrix
        start = self.start_location

        assert rack_capa == 2, "이 함수는 랙당 SKU 2개 조건을 전제로 합니다. rack_capacity=2로 설정해주세요."

        # 1. 카트-SKU 이진 매트릭스
        cart_sku_matrix = orders.pivot_table(index='CART_NO', columns='SKU_CD', aggfunc='size', fill_value=0)

        # 2. 카트 간 거리 → 유사도 → 클러스터링
        dist_matrix = pairwise_distances(cart_sku_matrix.values.astype(bool), metric='jaccard')
        labels = KMeans(n_clusters=24, random_state=42).fit_predict(dist_matrix)

        # 3. 클러스터별 SKU 집합
        orders['CLUSTER'] = orders['CART_NO'].map(dict(zip(cart_sku_matrix.index, labels)))
        cluster_skus = orders.groupby('CLUSTER')['SKU_CD'].apply(lambda x: list(set(x))).to_dict()

        # 4. 전체 SKU 빈도 및 SEQ
        sku_freq = orders['SKU_CD'].value_counts()
        sku_seq = orders.groupby('SKU_CD')['SEQ'].min().to_dict()

        assigned_skus = set()
        rack_assign_count = {rack: 0 for rack in racks}
        new_sku_to_loc = {}

        # 5. 클러스터별 대표 SKU → 대표 SKU 기준으로 배치
        for cluster, sku_list in cluster_skus.items():
            sku_list.sort(key=lambda x: (-sku_freq.get(x, 0), sku_seq.get(x, float('inf'))))
            if not sku_list:
                continue
            main_sku = sku_list[0]

            for rack in od_mat.loc[start, racks].sort_values().index:
                if rack_assign_count[rack] < rack_capa:
                    new_sku_to_loc[main_sku] = rack
                    assigned_skus.add(main_sku)
                    rack_assign_count[rack] += 1
                    base_rack = rack
                    break

            for sku in sku_list[1:]:
                if sku in assigned_skus:
                    continue
                for rack in od_mat.loc[base_rack, racks].sort_values().index:
                    if rack_assign_count[rack] < rack_capa:
                        new_sku_to_loc[sku] = rack
                        assigned_skus.add(sku)
                        rack_assign_count[rack] += 1
                        break

        # 6. 남은 SKU는 SLAP 방식
        for sku in sku_freq.index:
            if sku in assigned_skus:
                continue
            for rack in od_mat.loc[start, racks].sort_values().index:
                if rack_assign_count[rack] < rack_capa:
                    new_sku_to_loc[sku] = rack
                    assigned_skus.add(sku)
                    rack_assign_count[rack] += 1
                    break

        self.orders['LOC'] = self.orders['SKU_CD'].map(new_sku_to_loc)

        # ✅ 검증 1: SKU는 1 LOC만
        sku_loc_counts = self.orders.groupby('SKU_CD')['LOC'].nunique()
        if (sku_loc_counts > 1).any():
            raise ValueError("❌ SKU가 여러 LOC에 배치되었습니다.")

        # ✅ 검증 2: LOC에는 정확히 2 SKU
        loc_sku_counts = self.orders.groupby('LOC')['SKU_CD'].nunique()
        if not all(loc_sku_counts == 2):
            bad_locs = loc_sku_counts[loc_sku_counts != 2]
            raise ValueError(f"❌ LOC에 SKU가 정확히 2개가 아닙니다. (오류 LOC: {bad_locs.to_dict()})")

        print("✅ 클러스터 기반 재배치 완료")
        print("✅ 미배치 SKU 수:", len(set(orders['SKU_CD']) - set(new_sku_to_loc)))
        print("✅ 남은 빈 랙 수:", sum(v < rack_capa for v in rack_assign_count.values()))
    

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
        return total_time
    def calculate_average_picking_time(self, n_trials: int = 30) -> float:
        """
        초기 orders 상태를 유지하면서 solve() → calculate_total_picking_time()을 n_trials번 반복.
        평균 피킹 시간과 평균 알고리즘 소요 시간 출력.
        """
        total_times = []
        algo_times = []

        # 초기 orders 상태를 따로 저장
        original_orders = self.orders.copy()

        for i in range(n_trials):
            print(f"\n🔄 Trial {i+1}/{n_trials}")

            # 초기 상태로 복원
            self.orders = original_orders.copy()
            self._initialize_orders()  # LOC, CART_NO, SEQ 초기화

            # ⏱️ 알고리즘 시작
            start_time = time.time()
            self.solve()
            end_time = time.time()

            algo_elapsed = end_time - start_time

            # 피킹 시간 계산
            picking_time = self.calculate_total_picking_time()

            print(f"   ⏳ 알고리즘 시간: {algo_elapsed:.2f}초 | 총 피킹 시간: {picking_time:.2f}초")

            total_times.append(picking_time)
            algo_times.append(algo_elapsed)

        avg_picking_time = sum(total_times) / n_trials
        avg_algo_time = sum(algo_times) / n_trials

        print(f"\n🎯 {n_trials}회 평균 피킹 시간: {avg_picking_time:.2f}초")
        print(f"🎯 {n_trials}회 평균 알고리즘 시간: {avg_algo_time:.2f}초")

        return avg_picking_time

    def solve(self) -> pd.DataFrame:
        self.SLAP()
        self.OBSP_ORD_to_CART()
        self.OBSP_CART_to_BATCH()
        self.REMAPPING_CART()
        # self.reassign_sku_by_cart()
        self.reassign_sku_by_cart_similarity_clustered()
        self.OBSP_ORD_to_CART()
        self.OBSP_CART_to_BATCH()
        self.REMAPPING_CART()
        self.PRP()
        self.REMAPPING_CART()
        self.calculate_total_picking_time()
        return self.orders

def main(INPUT: pd.DataFrame, PARAMETER: pd.DataFrame, OD_MATRIX: pd.DataFrame) -> pd.DataFrame:
    solver = WarehouseSolver(INPUT, PARAMETER, OD_MATRIX)
    result = solver.solve()
    return result
'''average_time = solver.calculate_average_picking_time(n_trials=30)
    print(f"30회 평균 피킹 시간: {average_time:.2f}초")'''

if __name__ == "__main__":
    test_INPUT = pd.read_csv("./data/Sample_InputData.csv")
    test_PARAM = pd.read_csv("./data/Sample_Parameters.csv")
    test_OD = pd.read_csv("./data/Sample_OD_Matrix.csv", index_col=0, header=0)
    result = main(test_INPUT, test_PARAM, test_OD)
