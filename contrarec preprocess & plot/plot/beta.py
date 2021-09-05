import matplotlib.pyplot as plt
import numpy as np

beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

taobao_recall5 = [0.014875, 0.233297386, 0.318625817, 0.345995098, 0.316195261, 0.30385866, 0.300713235, 0.30227451, 0.302090686]
taobao_recall10 = [0.01491585, 0.242167484, 0.390982026, 0.439003268, 0.447035948, 0.446959967, 0.442752451, 0.439677288, 0.433523693]
taobao_ndcg5 = [0.012075986, 0.171961785, 0.209560459, 0.244254572, 0.228824778, 0.214430989, 0.211708127, 0.212619627, 0.215086489]
taobao_ndcg10 = [0.012087794, 0.174942558, 0.233486727, 0.274623433, 0.271694488, 0.261284358, 0.25816022, 0.257360981, 0.257742415]

tmall_recall5 = [0.005208333, 0.316255669, 0.296485261, 0.295103458, 0.293013039, 0.292800454, 0.292162698, 0.290320295, 0.290745465]
tmall_recall10 = [0.005208333, 0.43200822, 0.429350907, 0.435515873, 0.430945295, 0.432645975, 0.431795635, 0.43243339, 0.42903203]
tmall_ndcg5 = [0.005169104, 0.222401926, 0.192528306, 0.19102202, 0.185966424, 0.185735077, 0.185246734, 0.18395733, 0.182510139]
tmall_ndcg10 = [0.005169104, 0.260421329, 0.236497194, 0.237331579, 0.231542793, 0.231910182, 0.231433932, 0.230899371, 0.228184879]

x = np.arange(len(beta))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, taobao_ndcg5, width, label='NDCG@5', color='white', edgecolor='k', hatch='xxx')
rects2 = ax.bar(x + width/2, taobao_ndcg10, width, label='NDCG@10', color='white', edgecolor='k', hatch='***')
ax.plot(x, taobao_recall5, linestyle='-', color='k', marker='*', label='Recall@5')
ax.plot(x, taobao_recall10, linestyle=':', color='k', marker='o', label='Recall@10')
ax.set_xlabel('β')
ax.set_ylabel('Values')
ax.set_title('Values of Recall@K and NDCG@K on Dataset Taobao (K=5, 10)')
ax.set_xticks(x)
ax.set_xticklabels(beta)
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.92))

plt.show()
fig.savefig('beta_taobao.pdf', dpi=800, format='pdf')