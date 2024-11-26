import FinanceDataReader as fdr
from pandas import DataFrame
import numpy as np
import math
import scipy.spatial
import csv
import random


REPRESENTATIVE_STOCK_CODE = "005930" # 삼성전자


def load_stocks(start_date: str, end_date: str, max_count: int) -> list[DataFrame]:
    print("Loading stocks..")

    stocks: list[DataFrame] = []
    stock_infos = fdr.StockListing("KRX")[["Code", "Name"]].values

    correct_len = len(fdr.DataReader(REPRESENTATIVE_STOCK_CODE)[start_date : end_date])

    for code, name in stock_infos:
        stock = fdr.DataReader(code)[["Close", "Change", "Volume"]]

        if str(stock.index[0]) > start_date or str(stock.index[-1]) < end_date:
            continue

        stock = stock[start_date : end_date]

        if len(stock) != correct_len:
            continue
 
        if 0 in stock.Volume.values:
            continue

        stock.drop("Volume", axis=1)

        close_final = stock.Close[-1]
        change_final = stock.Change[-1]

        stock = stock[:-1]

        if math.isnan(stock.Change[0]):
            stock.Change[0] = 0
        
        stock.close_final = close_final
        stock.change_final = change_final
        stock.code = code
        stock.name = name
        stock.label = -1
        
        stocks.append(stock)
        if len(stocks) == max_count:
            break

    print(f"Loaded {len(stocks)} stocks\n")

    return stocks


def get_distance(x, y):
    corrcoef = np.corrcoef(x, y)[0][1]
    return math.sqrt(2 * (1 - corrcoef))


def get_similarity(x, y):
    dist = get_distance(x.Change.to_numpy().reshape((1, -1)), y.Change.to_numpy().reshape((1, -1)))
    return 1 / dist


def k_means_cluster(stocks: list[DataFrame], k: int, max_iter: int) -> list[list[DataFrame]]:
    print("K Means in progress...")

    centroids = np.stack([stock.Change.to_numpy().copy() for stock in random.sample(stocks, k=k)])
    clusters = [list[np.ndarray]() for _ in range(k)] 

    iter = 0
    while iter < max_iter:
        iter += 1
        no_change = True

        for stock in stocks:
            change_arr = stock.Change.to_numpy().reshape((1, -1))
            dist_matrix = scipy.spatial.distance.cdist(change_arr, centroids, get_distance)

            label = dist_matrix.argmin(axis=1)[0]
            if label != stock.label:
                no_change = False
                stock.label = label
            clusters[label].append(change_arr)

        if no_change:
            break

        for i in range(k):
            centroids[i] = np.mean(clusters[i], axis=0)
            clusters[i].clear()

    print(f"K Means terminated after {iter} iterations\n")

    stock_clusters = [list[DataFrame]() for _ in range(k)]
    for stock in stocks:
        stock_clusters[stock.label].append(stock)

    return stock_clusters


# TODO: Vectorization
def predict_cluster_final_close(cluster: list[DataFrame]):
    for stock in cluster:
        similarity_sum = 0
        aggregate_change = 0
        for other in cluster:
            if stock is other:
                continue

            similarity = get_similarity(stock, other)
            similarity_sum += similarity
            aggregate_change += similarity * other.Change[-1]

        if len(cluster) == 1:
            stock.pred_close_final = stock.Close[-1]
            stock.pred_change_final = stock.Change[-1]
        else:
            stock.pred_close_final = round(stock.Close[-1] + stock.Close[-1] * aggregate_change / similarity_sum) 
            stock.pred_change_final = aggregate_change / similarity_sum


def get_rmse(x: np.ndarray, y: np.ndarray):
    return np.sqrt(((x - y) ** 2).mean())


# last observed close value
def baseline_predict0(stock: DataFrame):
    return stock.Close[-7:].mean()

# average of values last 5 days
def baseline_predict1(stock: DataFrame):
    return stock.Close[-14:].mean()

START_DATE = "2023-03-02"
END_DATE = "2023-11-29"
MAX_STOCK_COUNT = 1000

K = 5
MAX_ITER = 500

stocks = load_stocks(START_DATE, END_DATE, MAX_STOCK_COUNT)

clusters = k_means_cluster(stocks, K, MAX_ITER)

print("Making predictions...")
for cluster in clusters:
    predict_cluster_final_close(cluster)
print("Finished making predictions\n")

pred = np.array([])
actual = np.array([])
baseline_pred0 = np.array([])
baseline_pred1 = np.array([])

for stock in stocks:
    pred = np.append(pred, stock.pred_close_final)
    actual = np.append(actual, stock.close_final)
    baseline_pred0 = np.append(baseline_pred0, baseline_predict0(stock))
    baseline_pred1 = np.append(baseline_pred1, baseline_predict1(stock))


rmse = get_rmse(pred, actual)
baseline_rmse0 = get_rmse(baseline_pred0, actual)
baseline_rmse1 = get_rmse(baseline_pred1, actual)

print(f"rmse: {rmse}")
print(f"baseline_rmse0: {baseline_rmse0}")
print(f"baseline_rmse1: {baseline_rmse1}")

pcc = np.corrcoef(pred, actual)[0][1]

f = open(f"out/{START_DATE}~{END_DATE}_주식개수={len(stocks)}_K={K}_예측 결과.csv", 'w', newline='')
f.write(f"pcc: {pcc}\n")
f.write(f"rmse: {rmse}\n")
f.write(f"baseline_rmse0: {baseline_rmse0}\n")
f.write(f"baseline_rmse1: {baseline_rmse1}\n")
w = csv.writer(f)
w.writerow(["Code", "Name", "Pred", "Actual", "Label"])
for stock in stocks:
    w.writerow([stock.code, stock.name, stock.pred_close_final, stock.close_final, stock.label])
f.close()