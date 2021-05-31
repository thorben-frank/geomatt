import numpy as np

from skorch import NeuralNet

from skorch.dataset import unpack_data
from skorch.utils import to_numpy


class FNeuralNet(NeuralNet):
    def __init__(self, module, beta=1., *args, **kwargs):
        super(FNeuralNet, self).__init__(module, *args, **kwargs)

        self.beta = beta

    def initialize(self, *args, **kwargs):
        super().initialize()
        self.with_forces = self.module_.with_forces
        self.beta_ = self.beta

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # y_pred = (E, F) w/ E.shape = [B,1] and F.shape = [B,N_v,3]
        # y_true = {"E": E_true, "F": F_true}
        E_pred, F_pred = y_pred

        if self.with_forces:
            F_true = y_true["F"]
            N_v = F_true.shape[1]
            force_loss = 1 / N_v * super().get_loss(y_pred=F_pred, y_true=F_true, X=X, training=training)
        else:
            assert len(F_pred) == 0
            force_loss = 0

        E_true = y_true["E"]
        energy_loss = super().get_loss(y_pred=E_pred, y_true=E_true, X=X, training=training)

        return self.beta_ * energy_loss + force_loss

    def predict_proba(self, X):
        energies = []
        forces = []

        #nonlin = self._get_predict_nonlinearity()
        for e, f in super().forward_iter(X, training=False):
            energies += [to_numpy(e)]
            forces += [to_numpy(f)]

        energies = np.concatenate(energies, 0)
        forces = np.concatenate(forces, 0)
        return {"E": energies, "F": forces}

    def validation_step(self, Xi, yi, **fit_params):
        #Xi, yi = unpack_data(batch)
        self.module_.eval()
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }