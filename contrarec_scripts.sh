#python main.py --cuda --dataset tmall --epoch_num 300 --trigger 50 --lr 0.00008
#python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.00008
#python main.py --cuda --dataset tmall --graph_encoder TAG --epoch_num 300 --trigger 50

# python main.py --cuda --graph_encoder GAT --beta 0.1
# python main.py --cuda --graph_encoder GAT --beta 0.2
# python main.py --cuda --graph_encoder GAT --beta 0.3
# python main.py --cuda --graph_encoder GAT --beta 0.4
# python main.py --cuda --graph_encoder GAT --beta 0.6
# python main.py --cuda --graph_encoder GAT --beta 0.7
# python main.py --cuda --graph_encoder GAT --beta 0.8
# python main.py --cuda --graph_encoder GAT --beta 0.9

# python main.py --cuda --graph_encoder TAG
# python main.py --cuda --graph_encoder AGNN
# python main.py --cuda --graph_encoder GIN
# python main.py --cuda --graph_encoder SG

# python main.py --cuda --graph_encoder GAT --fusion SUM
# python main.py --cuda --graph_encoder GAT --fusion MEAN
# python main.py --cuda --graph_encoder GAT --fusion PNLF

# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.1
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.2
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.3
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.4
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.6
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.7
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.8
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --lr 0.0001 --beta 0.9

# python main.py --cuda --dataset tmall --graph_encoder TAG --epoch_num 300 --trigger 50
# python main.py --cuda --dataset tmall --graph_encoder AGNN --epoch_num 300 --trigger 50
# python main.py --cuda --dataset tmall --graph_encoder GIN --epoch_num 300 --trigger 50
# python main.py --cuda --dataset tmall --graph_encoder SG --epoch_num 300 --trigger 50

# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --fusion SUM
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --fusion MEAN
# python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 50 --fusion PNLF