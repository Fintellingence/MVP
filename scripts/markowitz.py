import mvp
import matplotlib.pyplot as plt
import numpy as np
import math
#Defining DB path
path = '/home/naga/FintelligenceData/minute1_database_v1.db'
symbols = mvp.rawdata.get_db_symbols(path)
stock_space = mvp.portfolio_manager.RefinedSet(path, common_features="")


for symbol in symbols[:20]:
    stock_space.new_refined_symbol(symbol)

returns = dict()
for symbol in symbols[:20]:
    symbol_return = stock_space.refined_obj[symbol].get_returns(time_step='day')
    if len(symbol_return.index) == 1235:
        returns.update({symbol:symbol_return})

return_values = list()
for key in returns.keys():
    return_values.append(returns[key].values)
m = np.array(return_values)

cov_matrix = np.cov(m)
mean_returns = np.array(list(map(lambda x: x.mean(),return_values)))

#Generate Random Allocations
expected_returns = list()
expected_variance = list()
for i in range(0,10000):
    rand = np.random.rand(17)
    w = rand/rand.sum()
    expected_returns.append(np.dot(w,mean_returns))
    expected_variance.append(math.sqrt(np.matmul(np.matmul(w,cov_matrix),w)))

fig, ax = plt.subplots()
ax.scatter(expected_variance,expected_returns,s = 0.7)
ax.set_xlabel('Expected Risk')
ax.set_ylabel('Expected Return')
ax.legend()
plt.title('Random Portifolios')
plt.show()



