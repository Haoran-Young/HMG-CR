# python main.py --cuda --graph_encoder GCN --fusion MEAN --beta 0.4
# python main.py --cuda --graph_encoder SG --fusion MEAN --beta 0.4
# python main.py --cuda --graph_encoder TAG --fusion MEAN --beta 0.4
# python main.py --cuda --graph_encoder GIN --fusion MEAN --beta 0.4

# python main.py --cuda --dataset tmall --graph_encoder GCN --fusion MEAN --beta 0.2 --epoch_num 300 --trigger 100
# python main.py --cuda --dataset tmall --graph_encoder SG --fusion MEAN --beta 0.2 --epoch_num 300 --trigger 100
# python main.py --cuda --dataset tmall --graph_encoder TAG --fusion MEAN --beta 0.2 --epoch_num 300 --trigger 100
python main.py --cuda --dataset tmall --graph_encoder GIN --fusion MEAN --beta 0.2 --epoch_num 300 --trigger 150