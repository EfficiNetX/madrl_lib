"""
QMIX用のバッチデータ変換クラス
numpy配列からTorch Tensorへの変換を一元管理
"""
import torch
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class QMIXBatch:
    """
    QMIX学習用のバッチデータを格納するデータクラス
    型安全性を確保し、各フィールドの意味を明確化
    """
    share_obs: torch.Tensor  # (batch_size, episode_length + 1, share_obs_dim)
    obs: torch.Tensor  # (batch_size, episode_length + 1, n_agents, obs_dim)
    actions: torch.Tensor  # (batch_size, episode_length, n_agents, 1)
    rewards: torch.Tensor  # (batch_size, episode_length, n_agents, 1)
    dones: torch.Tensor  # (batch_size, episode_length, n_agents)
    mask: torch.Tensor  # (batch_size, episode_length)
    avail_actions: torch.Tensor  # (batch_size, episode_length + 1, n_agents, action_dim)


class BatchConverter:
    """
    numpy配列のバッチデータをTorch Tensorに変換する責務を持つクラス
    
    SOLID原則:
    - SRP: バッチ変換のみに責任を限定
    - OCP: 新しいデータ型追加時は_convert_fieldメソッドを拡張
    - DIP: 具体的なデバイスではなくargs.deviceに依存
    """
    
    def __init__(self, device):
        """
        Args:
            device: torch.device or str (e.g., 'cuda', 'cpu')
        """
        self.device = device
    
    def convert_episode_batch(self, episode_batch: Dict[str, np.ndarray]) -> QMIXBatch:
        """
        numpy配列のepisode_batchをQMIXBatchに変換
        
        Args:
            episode_batch: dict of np.ndarray
                - 'share_obs': (batch_size, episode_length + 1, share_obs_dim)
                - 'obs': (batch_size, episode_length + 1, n_agents, obs_dim)
                - 'actions': (batch_size, episode_length, n_agents, 1)
                - 'rewards': (batch_size, episode_length, n_agents)
                - 'dones': (batch_size, episode_length, n_agents)
                - 'mask': (batch_size, episode_length)
                - 'avail_actions': (batch_size, episode_length + 1, n_agents, action_dim)
        
        Returns:
            QMIXBatch: 全フィールドがTorch Tensorに変換されたバッチ
        """
        return QMIXBatch(
            share_obs=self._to_tensor(episode_batch["share_obs"], torch.float32),
            obs=self._to_tensor(episode_batch["obs"], torch.float32),
            actions=self._to_tensor(episode_batch["actions"], torch.long),
            rewards=self._to_tensor(episode_batch["rewards"], torch.float32),
            dones=self._to_tensor(episode_batch["dones"], torch.bool),
            mask=self._to_tensor(episode_batch["mask"], torch.bool),
            avail_actions=self._to_tensor(episode_batch["avail_actions"], torch.bool),
        )
    
    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """
        numpy配列をTorch Tensorに変換
        
        Args:
            array: 変換元のnumpy配列
            dtype: 変換後のTorch dtype
        
        Returns:
            torch.Tensor: 指定されたデバイスとdtypeのTensor
        """
        return torch.tensor(array, dtype=dtype, device=self.device)
