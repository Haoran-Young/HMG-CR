import torch

class Evaluator():
    def __init__(self, scores, labels):
        self.scores = scores
        self.labels = labels

        self.ks = [5, 10, 20, 50]

    def _recall(self, k):
        scores = self.scores.cpu()
        labels = self.labels.cpu()
        rank = (-scores).argsort(dim=1)
        cut = rank[:, :k]
        hit = labels.gather(1, cut)
        return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


    def _ndcg(self, k):
        scores = self.scores.cpu()
        labels = self.labels.cpu()
        rank = (-scores).argsort(dim=1)
        cut = rank[:, :k]
        hits = labels.gather(1, cut)
        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits.float() * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
        ndcg = dcg / idcg
        return ndcg.mean()


    def recalls_and_ndcgs_for_ks(self):
        metrics = {}

        scores = self.scores.cpu()
        labels = self.labels.cpu()
        answer_count = labels.sum(1)
        answer_count_float = answer_count.float()
        labels_float = labels.float()
        
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(self.ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

            position = torch.arange(2, 2+k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights).sum(1)
            idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
            ndcg = (dcg / idcg).mean().item()
            metrics['NDCG@%d' % k] = ndcg

        return metrics