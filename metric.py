from typing import List, Text

import torch
from pytorch_lightning.metrics import Metric

from dataset import SlotEncoding


# for debugging
def cat_labels(old: List[List[Text]], new: List[List[Text]]) -> List[List[Text]]:
    """
    Custom concatenation of lists to keep the
    state of the metric as lists of lists.
    """
    old.extend(new)
    return old


class JointAccuracy(Metric):
    def __init__(self, slot_encoding: SlotEncoding, dist_sync_on_step=False, eval: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.encoding = slot_encoding
        self.num_types = len(slot_encoding.type2int)
        self.eval = eval

        if self.eval:
            # to compute the joint accuracy per domain
            for domain in self.encoding.get_target_domains():
                self.add_state(f"joint_acc_numerator_{domain}", default=torch.tensor(0).float(), dist_reduce_fx="sum")
            # to compute the slot accuracy
            self.add_state("slot_acc_numerator", default=torch.tensor(0).float(), dist_reduce_fx="sum")
            self.add_state("slot_acc_denominator", default=torch.tensor(0).float(), dist_reduce_fx="sum")
            # for detailed prediction analysis
            self.add_state("guids", default=[], dist_reduce_fx=cat_labels)
            self.add_state("predictions", default=[], dist_reduce_fx=cat_labels)
            self.add_state("targets", default=[], dist_reduce_fx=cat_labels)

        # to compute the joint accuracy
        self.add_state("joint_acc_numerator", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("joint_acc_denominator", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, predictions: List[torch.Tensor], targets: torch.Tensor, guids: List[str]=None):
        """
        Update internal state with a new batch of predictions and targets.
        This function is called automatically by PyTorch Lightning.
        :param predictions: List[torch.Tensor] - list len: num_types, Tensor shape: [batch_size, max_turn_length]
            Value predicted by the model encoded as integer per slot type
        :param targets: Tensor, shape (batch_size, max_turn_length, num_types)
            Value ground truth per slot-type encoded as integers.
        """
        predictions = torch.cat(predictions, 2)
        accuracy = (predictions == targets).view(-1, self.num_types)

        if self.eval:
            self.slot_acc_numerator += torch.sum(accuracy).float()
            self.slot_acc_denominator += torch.sum(targets > -1).float()    # number of data

            # joint accuracy per domain
            for domain in self.encoding.get_target_domains():
                slot_ind = self.encoding.get_domain_type_code(domain)
                domain_accuracy = (predictions[:, :, slot_ind] == targets[:, :, slot_ind]).view(-1, len(slot_ind))
                try:
                    old_numerator = getattr(self, f"joint_acc_numerator_{domain}")
                except AttributeError:
                    self.add_state(
                        f"joint_acc_numerator_{domain}",
                        default=torch.tensor(0).float(),
                        dist_reduce_fx="sum"
                    )
                    old_numerator = getattr(self, f"joint_acc_numerator_{domain}")
                new_numerator = sum(torch.sum(domain_accuracy, 1) / len(slot_ind) == 1).float()
                setattr(self, f"joint_acc_numerator_{domain}", old_numerator + new_numerator)

            # for detailed prediction analysis
            predictions_list = predictions.view(-1, self.num_types).tolist()
            targets_list = targets.view(-1, self.num_types).tolist()
            true_predictions = [
                [self.encoding.get_value_name(i, p) for ((i, p), l) in zip(enumerate(pred), label) if l != -1]
                for pred, label in zip(predictions_list, targets_list)
            ]
            true_targets = [
                [self.encoding.get_value_name(i, l) for (p, (i, l)) in zip(pred, enumerate(label)) if l != -1]
                for pred, label in zip(predictions_list, targets_list)
            ]
            # Add predictions and labels to current state
            self.guids += [guid for guid in guids if guid != '']
            self.predictions += [p for p in true_predictions if p != []]
            self.targets += [t for t in true_targets if t != []]

        # join accuracy
        self.joint_acc_numerator += sum(torch.sum(accuracy, 1) / self.num_types == 1).float()
        self.joint_acc_denominator += torch.sum(targets[:, :, 0].view(-1) > -1, 0).float()  # number of turns

    def compute(self):
        joint_accuracy = self.joint_acc_numerator.float() / self.joint_acc_denominator.float()

        if self.eval:
            slot_accuracy = self.slot_acc_numerator.float() / self.slot_acc_denominator.float()
            perf = {
                'slots': {
                    'correct': self.slot_acc_numerator.item(),
                    'total': self.slot_acc_denominator.item(),
                    'slot_accuracy': slot_accuracy.item()
                },
                'turns': {
                    'correct': self.joint_acc_numerator.item(),
                    'total': self.joint_acc_denominator.item(),
                    'joint_accuracy': joint_accuracy.item()
                }
            }

            for domain in self.encoding.get_target_domains():
                domain_joint_accuracy = getattr(self, f"joint_acc_numerator_{domain}").float() / self.joint_acc_denominator.float()
                perf[f'turns_{domain}'] = {
                    'correct': getattr(self, f"joint_acc_numerator_{domain}").item(),
                    'total': self.joint_acc_denominator.item(),
                    'joint_accuracy': domain_joint_accuracy.item()
                }

            return {'performance': perf, 'predictions': self.predictions, 'targets': self.targets, 'guids': self.guids}

        else:
            return joint_accuracy
