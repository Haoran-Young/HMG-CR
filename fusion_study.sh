# python main.py --cuda --graph_encoder GAT --beta 0.4 --fusion SUM
# python main.py --cuda --graph_encoder GAT --beta 0.4 --fusion MLP
# python main.py --cuda --graph_encoder GAT --beta 0.4 --fusion PNLF

python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.2 --fusion SUM
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --beta 0.2 --fusion MLP
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --beta 0.2 --fusion PNLF