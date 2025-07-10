import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations
from collections import defaultdict
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

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
        Solve Storage Location Assignment Problem (SLAP) using MIP
        1. 빈도수+클러스터 기반 90% 배치(입출고 지점 Closeness)
        2. 주문별 총 거리 최소화 10%배치
        """
        rk = self.params.rack_capacity
        start_node = self.od_matrix.index[0]
        rack_columns = [col for col in self.od_matrix.columns if col.startswith('WP_')]
        
        # 1. SKU 간 동시 출현 행렬
        orders_by_ordno = self.orders.groupby('ORD_NO')['SKU_CD'].apply(list)
        sku_list = sorted(self.orders['SKU_CD'].unique())
        sku_index = {sku : idx for idx, sku in enumerate(sku_list)}
        co_occurence = np.zeros((len(sku_list), len(sku_list)), dtype= int)
        
        for sku_group in orders_by_ordno :
            for sku1, sku2 in combinations(sku_group, 2) :
                idx1, idx2= sku_index[sku1], sku_index[sku2]
                co_occurence[idx1][idx2] += 1 # 해당 sku의 다른 sku와의 등장 빈도
                co_occurence[idx2][idx1] += 1
            
            for sku in sku_group :
                idx = sku_index[sku]
                co_occurence[idx][idx] += 1 # 해당 sku의 등장 빈도

        co_matrix= pd.DataFrame(co_occurence, index= sku_list, columns= sku_list)
        
        # 2. 클러스터링
        cosine_dist = cosine_distances(co_matrix) # 자기 자신과의 유사도 1, 다른 sku와의 유사도 0~1
        clustering = AgglomerativeClustering(n_clusters= None, metric= 'precomputed', linkage= 'average', distance_threshold= 0.3)
        cluster_labels = clustering.fit_predict(cosine_dist)
        sku_cluster_map = pd.DataFrame({'SKU' : co_matrix.index, 'Cluster' : cluster_labels})
        
        print('총 생성된 클러스터 수 : ', sku_cluster_map['Cluster'].nunique())
        
        # 3. 클러스터 중 빈도 대표
        sku_freq = self.orders['SKU_CD'].value_counts().to_dict()
        cluster_freq = sku_cluster_map.copy()
        cluster_freq['Frequence'] = cluster_freq['SKU'].map(sku_freq)
        cluster_max_freq = cluster_freq.groupby('Cluster')['Frequence'].max().sort_values(ascending= False)
        
        # 4. 배치할 클러스터 비율 선택 - 90%
        top_n = int(len(cluster_max_freq) * 0.9) or 1
        top_clusters = cluster_max_freq.head(top_n).index.tolist()
        
        priority_skus = sku_cluster_map[sku_cluster_map['Cluster'].isin(top_clusters)]['SKU'].tolist()
        n_full_racks = len(priority_skus) // rk
        
        priority_skus_to_assign = priority_skus[:n_full_racks * rk] # 완전히 채울 수 있는 sku만
        remaining_skus = priority_skus[n_full_racks * rk :] # 나머지는 MIP
        
        # 5. 클러스터 입출고 지점 인접 랙에 우선 배치
        rack_distances = self.od_matrix.loc[start_node, rack_columns].sort_values()
        rack_iter = iter(rack_distances.index)
        rack_sku_alloctation = defaultdict(list)
        current_rack = next(rack_iter)
        
        for sku in priority_skus_to_assign :
            while len(rack_sku_alloctation[current_rack]) >= rk : 
                current_rack = next(rack_iter)
            rack_sku_alloctation[current_rack].append(sku)
        
        # 6. 결과
        priority_alloc_df = pd.DataFrame(
            [(sku, rack) for rack, skus in rack_sku_alloctation.items() for sku in skus],
            columns= ['SKU', 'Assigned_Rack']
        )
        
        # 7. 남은 SKU / 남은 랙에 MIP로 배치
        orders_by_ordno_dict = self.orders.groupby('ORD_NO')['SKU_CD'].apply(list).to_dict()
        
        used_skus = set(priority_alloc_df['SKU'])
        used_racks = set(priority_alloc_df['Assigned_Rack'])
        remaining_racks = sorted(set(rack_columns) - used_racks)
        
        x = {(sku, rack): LpVariable(f"x_{sku}_{rack}", cat=LpBinary) for sku in remaining_skus for rack in remaining_racks}
        z = {}
        
        prob = LpProblem("MIT_Optimization", LpMinimize)
        
        objective_terms = []
        
        for ord_skus in orders_by_ordno_dict.values():
            for sku in ord_skus:
                if sku in remaining_skus:
                    for rack in remaining_racks:
                        dist_start = self.od_matrix.loc["oWP_Start", rack]
                        objective_terms.append(dist_start * x[(sku, rack)])
                elif sku in used_skus:
                    rack = priority_alloc_df[priority_alloc_df["SKU"] == sku]["Assigned_Rack"].values[0]
                    dist_start = self.od_matrix.loc["oWP_Start", rack]
                    objective_terms.append(dist_start)

            for i in range(len(ord_skus)):
                for j in range(i + 1, len(ord_skus)):
                    sku_i, sku_j = ord_skus[i], ord_skus[j]
                    for rack_i in remaining_racks:
                        for rack_j in remaining_racks:
                            if rack_i == rack_j:
                                continue
                            dist = self.od_matrix.loc[rack_i, rack_j]
                            if sku_i in remaining_skus and sku_j in remaining_skus:
                                z_var = LpVariable(f"z_{sku_i}_{rack_i}_{sku_j}_{rack_j}", cat=LpBinary)
                                z[(sku_i, rack_i, sku_j, rack_j)] = z_var
                                prob += z_var <= x[(sku_i, rack_i)]
                                prob += z_var <= x[(sku_j, rack_j)]
                                prob += z_var >= x[(sku_i, rack_i)] + x[(sku_j, rack_j)] - 1
                                objective_terms.append(dist * z_var)
                            elif sku_i in remaining_skus and sku_j in used_skus:
                                rack_j_fixed = priority_alloc_df[priority_alloc_df["SKU"] == sku_j]["Assigned_Rack"].values[0]
                                dist = self.od_matrix.loc[rack_i, rack_j_fixed]
                                objective_terms.append(dist * x[(sku_i, rack_i)])
                            elif sku_j in remaining_skus and sku_i in used_skus:
                                rack_i_fixed = priority_alloc_df[priority_alloc_df["SKU"] == sku_i]["Assigned_Rack"].values[0]
                                dist = self.od_matrix.loc[rack_i_fixed, rack_j]
                                objective_terms.append(dist * x[(sku_j, rack_j)])

            for sku in ord_skus:
                if sku in remaining_skus:
                    for rack in remaining_racks:
                        dist_end = self.od_matrix.loc[rack, "oWP_End"]
                        objective_terms.append(dist_end * x[(sku, rack)])
                elif sku in used_skus:
                    rack = priority_alloc_df[priority_alloc_df["SKU"] == sku]["Assigned_Rack"].values[0]
                    dist_end = self.od_matrix.loc[rack, "oWP_End"]
                    objective_terms.append(dist_end)

        prob += lpSum(objective_terms)

        # 제약 조건
        for sku in remaining_skus:
            prob += lpSum(x[(sku, rack)] for rack in remaining_racks) == 1  # 한 SKU는 한 랙에만

        for rack in remaining_racks:
            prob += lpSum(x[(sku, rack)] for sku in remaining_skus) <= rk  # 랙당 수용 한도

        solver = PULP_CBC_CMD(timeLimit=100, msg=1)  # 1% GAP 도달 시 중단
        prob.solve(solver)

        mit_result = {(sku, rack): var.value() for (sku, rack), var in x.items() if var.value() == 1}
        mit_alloc_df = pd.DataFrame([(sku, rack) for (sku, rack), v in mit_result.items()],
                                    columns=["SKU", "Assigned_Rack"])

        final_alloc_df = pd.concat([priority_alloc_df, mit_alloc_df], ignore_index=True)
        
        sku_to_location = dict(zip(final_alloc_df['SKU'], final_alloc_df['Assigned_Rack']))
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