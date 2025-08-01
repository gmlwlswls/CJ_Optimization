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
# PRP - KNN

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
        배치 단위로 피커-카트 배정을 수행한다.
        Higest Workload cart를 기준으로 유사도가 먼 cart들의 조합을 만들어 anchor 작업량과 합이 같도록 더미를 구성한다.
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

            # anchor cart를 picker1에 할당
            pickers[1] = [highest_workload_cartno]
            batch.append(highest_workload_cartno)
            assigned_carts.add(highest_workload_cartno)

            # anchor와 유사도 계산
            anchor_orders = self.orders[self.orders['CART_NO'] == highest_workload_cartno]
            anchor_skus = anchor_orders['SKU_CD'].unique() # unique하는게 맞을듯! SKU_CD의 갯수 문제가 아니라 SKU_CD 종류로 인해 발생하는 작업량이 크게 늘어나는거라

            cart_similarities = []
            for _, row in unassigned.iterrows():
                if row['CART_NO'] == highest_workload_cartno :
                    continue
                candidate_orders = self.orders[self.orders['CART_NO'] == row['CART_NO']]
                candidate_skus = candidate_orders['SKU_CD'].unique()
                intersection = len(set(anchor_skus) & set(candidate_skus))
                union = len(set(anchor_skus) | set(candidate_skus))
                jaccard_sim = intersection / union if union > 0 else 0
                cart_similarities.append((row['CART_NO'], jaccard_sim, row['WORKLOAD']))

            # 자카드 유사도를 기준으로 오름차순
            # 작업량 기준으로 내림차순
            # 유사도가 낮은 카트(피커별로 유사도가 낮게 배정해야까) + 유사도가 같다면 작업량이 높은 카트 순서
            cart_similarities.sort(key=lambda x: (x[1], -x[2]))
 
            # 작업량 가장 높은 카트 제외한 CART_NO, WORKLOAD만 가져오기
            # [(59, 18), (76, 18), (82, 18), (112, 18), (3, 17), ,,]
            remaining_carts = [(c[0], c[2]) for c in cart_similarities] 
            used_carts = set()

            # picker_id > 2,3,4
            for picker_id in range(2, n_pickers + 1):
                best_combo = None
                best_diff = float('inf')

                # 1~3개 조합 탐색
                # 궁금증) 조합은 제한이 없어도 되지 않는지?
                for r in range(1, min(4, len(remaining_carts) + 1)):
                    # combo : (CART_NO, WORKLOAD, ) > 1개짜리 조합 ~ 3개짜리 조합
                    for combo in combinations(remaining_carts, r):
                        combo_carts = [c[0] for c in combo]
                        combo_workload = sum(c[1] for c in combo)

                        if any(c in used_carts for c in combo_carts):
                            continue

                        # 정해진 한 카트와 작업량 차이가 가장 작은 카트 합
                        diff = abs(combo_workload - highest_workload_quantity)
                        if diff < best_diff:
                            best_diff = diff
                            best_combo = combo_carts

                        if diff == 0:
                            break
                    if best_diff == 0:
                        break
                
                # best_cartno 조합
                if best_combo:
                    pickers[picker_id] = best_combo # {2 : 작업량 차이 제일 작은 조합}
                    batch.extend(best_combo)
                    used_carts.update(best_combo)
                    assigned_carts.update(best_combo)

            # 더미 기록
            for picker, carts in pickers.items():
                for cart in carts:
                    self.orders.loc[self.orders['CART_NO'] == cart, 'PICKER_NO'] = picker
                    self.orders.loc[self.orders['CART_NO'] == cart, 'BATCH_NO'] = batch_no
                    # BATCH_NO : 카트들을 피커수에 맞게 묶은 배치 1(카트 N개가 피커 4명에게 배치되어 있음)

            batch_no += 1

    def solve_picker_routing_KNN(self) -> None:
        """
        PRP: 각 BATCH_NO → PICKER_NO → CART_NO 순으로,
        CART 안의 SKU LOC를 OD_MATRIX를 활용해 가까운 순으로 방문하도록 SEQ 지정
        """
        seq_records = []

        # 그룹 단위로 처리
        for (batch_no, picker_no, cart_no), group in self.orders.groupby(['BATCH_NO', 'PICKER_NO', 'CART_NO']):
            locs = group['LOC'].unique()
            remaining_locs = set(locs)
            route = []
            current_loc = self.start_location

            while remaining_locs:
                next_loc = min(
                    remaining_locs,
                    key=lambda loc: self.od_matrix.loc[current_loc, loc]
                )
                route.append(next_loc)
                remaining_locs.remove(next_loc)
                current_loc = next_loc

            # LOC 방문 순서를 기준으로 SKU 행에 SEQ를 매김
            seq = 1
            for loc in route:
                loc_rows = group[group['LOC'] == loc]
                for idx in loc_rows.index:
                    seq_records.append((idx, seq))
                    seq += 1

        # SEQ를 orders에 반영
        seq_df = pd.DataFrame(seq_records, columns=['index', 'SEQ']).set_index('index')
        self.orders.loc[seq_df.index, 'SEQ'] = seq_df['SEQ']


    def solve(self) -> pd.DataFrame:
        """Execute complete warehouse optimization solution"""
        self.solve_storage_location()
        self.solve_order_batching()
        self.assign_batches_combination_sum_strategy()
        self.solve_picker_routing_KNN()
        '''if self.orders['LOC'].isna().any():
            raise ValueError("LOC에 할당되지 않은 SKU가 있습니다.")
        if self.orders['CART_NO'].isna().any():
            raise ValueError("CART_NO가 지정되지 않았습니다.")
        if self.orders['SEQ'].isna().any():
            raise ValueError("SEQ가 지정되지 않았습니다.")'''
        return self.orders

def main(INPUT: pd.DataFrame, PARAMETER: pd.DataFrame, OD_MATRIX: pd.DataFrame) -> pd.DataFrame:
    solver = WarehouseSolver(INPUT, PARAMETER, OD_MATRIX)
    return solver.solve()

if __name__ == "__main__":
    import time
    test_INPUT = pd.read_csv("sample_data/InputData.csv")
    test_PARAM = pd.read_csv("sample_data/Parameters.csv")
    test_OD = pd.read_csv("sample_data/OD_Matrix.csv", index_col=0, header=0)
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