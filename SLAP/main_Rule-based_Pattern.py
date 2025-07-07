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
        # 단일/동시 주문 경향 계산
        order_counts = self.orders.groupby('ORD_NO')['SKU_CD'].nunique()
        multi_orders = order_counts[order_counts > 1].index

        sku_total_orders = self.orders.groupby('SKU_CD')['ORD_NO'].nunique()
        sku_multi_orders = self.orders[self.orders['ORD_NO'].isin(multi_orders)].groupby('SKU_CD')['ORD_NO'].nunique()

        sim_prop = sku_multi_orders / sku_total_orders
        sim_prop = sim_prop.fillna(0)
        single_prop = 1 - sim_prop

        sku_freq = sku_total_orders

        weight = []
        for sku in sku_freq.index:
            f = sku_freq[sku]
            s = single_prop[sku]
            m = sim_prop[sku]
            w = s * f if s >= 0.5 else m * f
            weight.append((sku, s, m, f, w))

        df = pd.DataFrame(weight, columns=['SKU', '단일주문경향', '동시주문경향', '주문빈도', '총가중치'])
        df = df.sort_values(by='총가중치', ascending=False).reset_index(drop=True)

        # 입출고/경유지 기준 거리 정렬
        valid_racks = [rack for rack in self.od_matrix.columns if not rack.startswith("oWP_")]
        
        closeness_racks = [r for r in self.od_matrix.loc[self.start_location].sort_values().index.tolist() if r in valid_racks]
        betweenness_racks = [r for r in self.od_matrix.loc[self.end_location].sort_values().index.tolist() if r in valid_racks]

        all_racks = list(dict.fromkeys(closeness_racks + betweenness_racks))  # 중복 제거 유지
        sku_to_location = {}
        rack_cursor = 0

        for _, row in df.iterrows():
            sku = row['SKU']
            if row['단일주문경향'] >= 0.5:
                # 단일 주문 경향이 0.5 이상이면 입출고지 기준 가까운 랙 배치
                for loc in closeness_racks:
                    if list(sku_to_location.values()).count(loc) < self.params.rack_capacity:
                        sku_to_location[sku] = loc
                        break
            else:
                # 동시 주문 경향이 더 크면 경유지 기준 가까운 랙 배치
                for loc in betweenness_racks:
                    if list(sku_to_location.values()).count(loc) < self.params.rack_capacity:
                        sku_to_location[sku] = loc
                        break

        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)

    def solve_order_batching(self) -> None:
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
        self.orders = self.orders.sort_values(['CART_NO', 'LOC'])
        self.orders['SEQ'] = self.orders.groupby('CART_NO').cumcount() + 1

    def solve(self) -> pd.DataFrame:
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