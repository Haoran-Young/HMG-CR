import numpy as np
import matplotlib.pyplot as plt

taobao_folder = 'taobao'
taobao_c_loss_path = taobao_folder+'/contrastive_loss.csv'
taobao_r_loss_path = taobao_folder+'/rec_loss.csv'

taobao_c_loss = np.loadtxt(taobao_c_loss_path)
taobao_r_loss = np.loadtxt(taobao_r_loss_path)

taobao_epoch = [i for i in range(50)]
taobao_c_loss = taobao_c_loss[: len(taobao_epoch)]
taobao_r_loss = taobao_r_loss[: len(taobao_epoch)]

tmall_folder = 'tmall'
tmall_c_loss_path = tmall_folder+'/contrastive_loss.csv'
tmall_r_loss_path = tmall_folder+'/rec_loss.csv'

tmall_c_loss = np.loadtxt(tmall_c_loss_path)
tmall_r_loss = np.loadtxt(tmall_r_loss_path)

tmall_epoch = [i for i in range(250)]
tmall_c_loss = tmall_c_loss[: len(tmall_epoch)]
tmall_r_loss = tmall_r_loss[: len(tmall_epoch)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

ax1.plot(taobao_epoch, taobao_c_loss, label='contrastive', linestyle='-', color='k')
ax1.plot(taobao_epoch, taobao_r_loss, label='recommendation', linestyle=':', color='k')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.set_title('Training Loss on Taobao')
ax1.legend(loc='best', fontsize=8)

ax2.plot(tmall_epoch, tmall_c_loss, label='contrastive', linestyle='-', color='k')
ax2.plot(tmall_epoch, tmall_r_loss, label='recommendation', linestyle=':', color='k')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.set_title('Training Loss on Tmall')
ax2.legend(loc='best', fontsize=8)

plt.show()
fig.savefig('training_loss.pdf', dpi=800, format='pdf')