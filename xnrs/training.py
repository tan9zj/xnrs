import torch
import torch.nn as nn
from requests.packages import target
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import wandb
from abc import ABC, abstractmethod
import os
from os.path import join, exists
from pandas import DataFrame
import datetime
from typing import Optional

from .evaluation import metrics as eval
from . import utils
from .utils import batch_to_device
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class BaseTrainer(ABC):

    def __init__(self, 
            cfg: DictConfig, 
            model: nn.Module, 
            trainset: Dataset, 
            testset: Dataset
        ):
        self.cfg = cfg
        self.model = model
        self._init_dataloaders(trainset, testset)

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self._init_loss()

        self.current_epoch = 0
        self.current_train_step = 0
        self.current_test_step = 0

    @abstractmethod
    def _init_loss(self):
        raise NotImplementedError(
            'BaseTrainer needs to be subclassed with a loss implementation'
        )

    def _init_dataloaders(self, trainset: Dataset, testset: Dataset):
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle_data,
            drop_last=True,
            num_workers=self.cfg.num_workers
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0  # self.cfg.num_workers
        )

    def forward(self, batch) -> torch.Tensor:
        '''implement output activation here if needed'''
        pass
    
    def _save_checkpoint(self):
        print('saving checkpoint')
        ckpt_path = join(self.cfg.dir, self.cfg.name, 'checkpoints')
        if not exists(ckpt_path):
            os.makedirs(ckpt_path)
        ckpt_dict = {
            'config': dict(self.cfg),
            'model_name': self.cfg.name,
            'state_dict': self.model.state_dict()
        }
        torch.save(ckpt_dict, join(ckpt_path, f'ckpt_{self.current_epoch}'))

    def _save_scores(self, targets: np.array, scores: np.array, stats: Optional[dict] = None):
        print('saving scores')
        scores_path = join(self.cfg.dir, self.cfg.name, 'predictions')
        if not exists(scores_path):
            os.makedirs(scores_path)
        scores_dict = {
            'targets': targets,
            'scores': scores,
            'stats': stats
        }
        torch.save(scores_dict, join(scores_path, f'predictions_{self.current_epoch}'))

    def _train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()
        t = batch['targets'].to(self.device)
        s = self.forward(batch)
        if 'weights' in batch.keys():
            w = batch['weights'].to(self.device)
            loss = self.L(s, t, w)
        else:
            loss = self.L(s, t)
        loss.backward()
        self.optimizer.step()
        results = {
            'loss': loss.detach().cpu().numpy(),
            'logits': s
        }
        return results

    @abstractmethod
    def _test_step(self, batch: dict) -> dict:
        raise NotImplementedError('BaseTrainer needs to be subclassed with a _test_step implementation')

    def _train_iteration(self) -> dict:
        self.model.train()
        iteration_results = []
        for batch in tqdm(self.trainloader):
            results = self._train_step(batch)
            iteration_results.append(results)
            self._after_train_step(results)
            if self.cfg.debug:
                print('debugging - interrupting after first step')
                break
        self._after_train_iteration(iteration_results)
        return iteration_results

    def _test_iteration(self) -> list:
        self.model.eval()
        iteration_results = []
        for batch in tqdm(self.testloader):
            results = self._test_step(batch)
            self._after_test_step(results)
            iteration_results.append(results)
            if self.cfg.debug:
                print('debugging - interrupting after first step')
                break
        self._after_test_iteration(iteration_results)
        return iteration_results

    def train(self):
        # TODO: evetually switch to logging and testing in step-wise periods?
        for e in range(self.cfg.n_epochs):
            self.current_epoch = e
            print(f'\n Epoch {e}:')
            print('training:')
            train_results = self._train_iteration()
            if (e + 1) % self.cfg.test_freq == 0\
                or e == self.cfg.n_epochs - 1:
                print('testing:')
                test_results = self._test_iteration()
            if self.cfg.debug:
                print('debugging - interrupting after first epoch')
                break
        if self.cfg.n_epochs == 0:
            test_results = self._test_iteration()
        # self._after_training(
        #     last_train_results=train_results,
        #     last_test_results=test_results
        # )

    def _after_train_step(self, results: dict):
        pass

    def _after_test_step(self, results: dict):
        pass

    def _after_train_iteration(self, results: dict):
        if self.cfg.ckpt_freq is not None:
            if self.current_epoch % self.cfg.ckpt_freq == 0\
                or self.current_epoch == self.cfg.n_epochs - 1:
                self._save_checkpoint()
        epoch_loss = np.mean([d['loss'] for d in results])
        if self.cfg.wandb:
            wandb.log({
                'train_loss': epoch_loss, 
                'epoch': self.current_epoch
                })
        print('train loss: ', epoch_loss)

    def _after_test_iteration(self, results: list):
        pass

    def _after_training(self, last_train_results: list, last_test_results: list):
        pass


class RankingTrainer(BaseTrainer):

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        # TODO combine this with train step in forward
        t = batch['targets'].to(self.device)
        s = self.forward(batch)
        if 'weights' in batch.keys():
            w = batch['weights'].to(self.device)
            loss = self.L(s, t, w)
        else:
            loss = self.L(s, t)
        loss = loss.cpu().numpy()
        t = t.cpu().squeeze().numpy()
        s = s.cpu().squeeze().numpy()
        return_dict = {
            'ndcg@5': eval.ndcg_score(t, s, k=5),
            'ndcg@10': eval.ndcg_score(t, s, k=10),
            'rr': eval.rr_score(t, s),
            'ctr@1': eval.ctr_score(t, s, k=1),
            'auc': eval.auc_score(t, s),
            'acc': eval.acc_score(t, s),
            'rec': eval.recall_score(t, s), 
            'prec': eval.precision_score(t, s),
            'conf': eval.confusion_matrix(t, s),
            'scores': s,
            'targets': t,
            'loss': loss
        }
        self._after_test_step(return_dict)
        return return_dict

    def _after_test_iteration(self, results: list):
        loss = np.array([d['loss'] for d in results])
        ndcg5 = np.array([d['ndcg@5'] for d in results])
        ndcg10 = np.array([d['ndcg@10'] for d in results])
        rr = np.array([d['rr'] for d in results])
        ctr = np.array([d['ctr@1'] for d in results])
        acc = np.array([d['acc'] for d in results])
        auc = np.array([d['auc'] for d in results])
        rec = np.array([d['rec'] for d in results])
        prec = np.array([d['prec'] for d in results])
        epoch_loss = np.mean(loss)
        epoch_ndcg5 = np.mean(ndcg5)
        epoch_ndcg10 = np.mean(ndcg10)
        epoch_mrr = np.mean(rr)
        epoch_acc = np.mean(acc)
        epoch_auc = np.mean(auc)
        epoch_rec = np.mean(rec)
        epoch_prec = np.mean(prec)
        epoch_ctr = np.mean(ctr)
        epoch_conf = np.sum([d['conf'] for d in results], axis=0)
        scores = np.concatenate([d['scores'] for d in results])
        targets = np.concatenate([d['targets'] for d in results])
        score_hist = wandb.Histogram(scores, num_bins=50)
        conf_table = wandb.Table(
            columns=['0', '1'],
            data=epoch_conf.tolist()
            )
        self._save_scores(
            targets=targets, 
            scores=scores,
            stats={
                'auc': auc,
                'mrr': rr,
                'ndcg@5': ndcg5,
                'ndcg@10': ndcg10
            })
        if self.cfg.wandb:
            wandb.log({
                'test_loss': epoch_loss,
                'ndcg@5': epoch_ndcg5,
                'ndcg@10': epoch_ndcg10,
                'mrr': epoch_mrr,
                'ctr@1': epoch_ctr,
                'auc': epoch_auc,
                'acc': epoch_acc,
                'rec': epoch_rec,
                'prec': epoch_prec,
                'conf': conf_table,
                'scores': score_hist,
                'epoch': self.current_epoch
                })
        print(f'test loss: {epoch_loss:.4f}, ndcg@5: {epoch_ndcg5:.4f}, '\
                + f'mrr: {epoch_mrr:.4f}, ctr@1: {epoch_ctr:.4f}, auc: {epoch_auc:.4f}, '\
                + f'acc: {epoch_acc:.4f}, rec: {epoch_rec:.4f}, prec: {epoch_prec:.4f}'
            )


class BCERankingTrainer(RankingTrainer):

    def _init_loss(self):
        self.L = nn.BCELoss()

    def forward(self, batch) -> torch.Tensor:
        '''implement output activation here if needed'''
        return torch.sigmoid(self.model.forward(batch))


class BCELogitsRankingTrainer(RankingTrainer):

    def _init_loss(self):
        self.L = nn.functional.binary_cross_entropy_with_logits

    def forward(self, batch) -> torch.Tensor:
        '''no output activation applied here'''
        batch_to_device(batch, self.device)
        return self.model.forward(batch)

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        '''need to add sigmoid after loss computation here'''
        t = batch['targets'].to(self.device)
        s = self.forward(batch)
        if 'weights' in batch.keys():
            w = batch['weights'].to(self.device)
            loss = self.L(s, t, w)
        else:
            loss = self.L(s, t)
        loss = loss.cpu().numpy()
        t = t.squeeze().cpu().numpy()
        # adding sigmoid to compute test metrics
        s = torch.sigmoid(s).squeeze().cpu().numpy()
        return_dict = {
            'ndcg@5': eval.ndcg_score(t, s, k=5),
            'ndcg@10': eval.ndcg_score(t, s, k=10),
            'rr': eval.rr_score(t, s),
            'ctr@1': eval.ctr_score(t, s, k=1),
            'auc': eval.auc_score(t, s),
            'acc': eval.acc_score(t, s),
            'rec': eval.recall_score(t, s), 
            'prec': eval.precision_score(t, s),
            'conf': eval.confusion_matrix(t, s),
            'scores': s,
            'targets': t,
            'loss': loss
        }
        self._after_test_step(return_dict)
        return return_dict


class MSERankingTrainer(RankingTrainer):
    
    def _init_loss(self):
        def loss(prediction, target, weight: Optional[torch.tensor] = None):
            if weight is not None:
                l = nn.functional.mse_loss(prediction, target, reduction='none')
                l = l * weight
                return torch.mean(l)
            else:
                return nn.functional.mse_loss(prediction, target)
        self.L = loss

    def forward(self, batch) -> torch.Tensor:
        '''implement output activation here if needed'''
        batch_to_device(batch, self.device)
        oupt = self.model.forward(batch)
        return torch.relu(oupt)
        # return self.model.forward(batch)

class ContrastiveRankingTrainer(MSERankingTrainer):

    def _init_loss(self):
        super()._init_loss()
        self.temperature = self.cfg.get('contrastive_temperature', 0.1)
        self.lambda_cl = self.cfg.get('contrastive_lambda', 0.1)

    def _train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()

        targets = batch['targets'].to(self.device)
        preds = self.forward(batch)
        loss_rec = self.L(preds, targets)

        user_embeddings = self.model.get_user_embeddings(batch) # shape: [B, D]

        main_cats = batch['main_category']
        unique_cats = list(set(main_cats))
        cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
        cat_labels = torch.tensor([cat_to_idx[c] for c in main_cats]).to(self.device)  # shape: [B]

        # CL loss
        loss_cl = self._compute_contrastive_loss(user_embeddings, cat_labels)

        loss_total = loss_rec + self.lambda_cl * loss_cl
        loss_total.backward()
        self.optimizer.step()

        return {
            "loss": loss_total.item(),
            "loss_rec": loss_rec.item(),
            "loss_cl": loss_cl.item(),
            "logits": preds
        }

    def _compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
            embeddings: B,D
            labels: B
        """
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        B = embeddings.size(0)
        loss = 0.0
        count = 0
        device = labels.device

        for i in range(B):
            # if same cat label -> positive
            # different cat label -> negative
            pos_mask = (labels == labels[i]) & (torch.arange(B, device=device) != i)
            neg_mask = (labels != labels[i])

            pos_sims = sim_matrix[i][pos_mask] / self.temperature
            all_sims = sim_matrix[i][torch.arange(B, device=device) != i] / self.temperature
            if len(pos_sims) == 0:
                continue
            numerator = torch.exp(pos_sims).sum()
            denominator = torch.exp(all_sims).sum()

            loss += -torch.log(numerator / (denominator + 1e-12))
            count += 1
        return loss / (count + 1e-8)

    def _after_train_step(self, results: dict):
        if self.cfg.wandb:
            wandb.log({
                "loss_total": results["loss"],
                "loss_rec": results["loss_rec"],
                "loss_cl": results["loss_cl"],
                "step": self.current_train_step
            })
        self.current_train_step += 1

    def _after_train_iteration(self, results: list):
        if self.cfg.ckpt_freq is not None:
            if self.current_epoch % self.cfg.ckpt_freq == 0 or self.current_epoch == self.cfg.n_epochs - 1:
                self._save_checkpoint()

        epoch_loss_total = np.mean([d['loss'] for d in results])
        epoch_loss_rec = np.mean([d['loss_rec'] for d in results])
        epoch_loss_cl = np.mean([d['loss_cl'] for d in results])

        if self.cfg.wandb:
            wandb.log({
                'epoch_loss_total': epoch_loss_total,
                'epoch_loss_rec': epoch_loss_rec,
                'epoch_loss_cl': epoch_loss_cl,
                'epoch': self.current_epoch
            })

        print(f"[Epoch {self.current_epoch}] total: {epoch_loss_total:.4f}, "
              f"rec: {epoch_loss_rec:.4f}, cl: {epoch_loss_cl:.4f}")

    # def _visualize_user_embeddings(self, results):
    #     all_embeds = []
    #     all_labels = []
    #
    #     for r in results:
    #         if 'user_embeddings' in r and 'main_category' in r:
    #             all_embeds.append(r['user_embeddings'])  # shape: [D]
    #             all_labels.append(r['main_category'])
    #
    #     if len(all_embeds) < 10:
    #         print("Too few embeddings for visualization.")
    #         return
    #
    #     embeds = np.stack(all_embeds)
    #     tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
    #     reduced = tsne.fit_transform(embeds)
    #
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     for cat in set(all_labels):
    #         idxs = [i for i, c in enumerate(all_labels) if c == cat]
    #         ax.scatter(reduced[idxs, 0], reduced[idxs, 1], label=cat, alpha=0.6)
    #     ax.legend()
    #     ax.set_title("User Embedding t-SNE")
    #
    #     if self.cfg.wandb:
    #         wandb.log({"user_embedding_tsne": wandb.Image(fig)})
    #     plt.close(fig)

    def _after_test_iteration(self, results: list):
        loss = np.mean([d['loss'] for d in results])
        ndcg5 = np.mean([d['ndcg@5'] for d in results])
        ndcg10 = np.mean([d['ndcg@10'] for d in results])
        auc = np.mean([d['auc'] for d in results])

        if self.cfg.wandb:
            wandb.log({
                'val_loss': loss,
                'val_ndcg@5': ndcg5,
                'val_ndcg@10': ndcg10,
                'val_auc': auc,
                'epoch': self.current_epoch
            })

        print(f"[Eval {self.current_epoch}] loss: {loss:.4f}, ndcg@5: {ndcg5:.4f}, auc: {auc:.4f}")

        if self.cfg.get("visualize_user_embeddings", False):
            self._visualize_user_embeddings(results)
