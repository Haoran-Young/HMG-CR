python main.py --cuda --graph_encoder GAT --beta 0.1 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.2 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.3 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.4 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.6 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.7 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.8 --fusion MEAN
python main.py --cuda --graph_encoder GAT --beta 0.9 --fusion MEAN

python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.1 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.2 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.3 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.4 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.6 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.7 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.8 --fusion MEAN
python main.py --cuda --dataset tmall --graph_encoder GAT --epoch_num 300 --trigger 100 --beta 0.9 --fusion MEAN