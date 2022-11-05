
import torch.nn as nn
from utils import *
from matplotlib.patches import Rectangle, Circle



class transfer_matrix(nn.Module):
    def __init__(self, mat_lst=['air', 'TiO2', 'MgF2', 'TiO2', 'Ag', 'MgF2', 'TiO2', 'SiO2'],
                 wlmin=300, wlmax=2500, num=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        initialize the transfer matrix class
        :param mat_lst: a list of materials, include the semi-inf air and include the semi-inf sub
        :param wlmin: min wavelength, unit: nm
        :param wlmax: max wavelength, unit: nm
        :param num: wavelength data point number
        """
        super(transfer_matrix, self).__init__()
        self.mat_lst = mat_lst  # layer material list
        self.n_mat = len(mat_lst)  # number of layers
        self.wlmin = wlmin
        self.wlmax = wlmax
        self.num = num
        self.wl = torch.linspace(wlmin, wlmax, num)  # wavelength list
        self.mat_n0, self.mat_k0 = self.get_index_spec()  # get n0 and k0 for all wavelengths and all layers
        self.target = target_trans(wlmin=wlmin, wlmax=wlmax, num=num, plot=False)
        self.solar = solar(wlmin=wlmin, wlmax=wlmax, num=num, plot=False)

    def get_index_spec(self):
        """
        get the refractive index matrix, row for different wavelength, col for different layer
        :return: two (real / imag) index matrix with dimension [num, len(mat_lst)]
        """
        n_mat = len(self.mat_lst)
        mat_n0 = torch.ones(self.num, n_mat)
        mat_k0 = torch.zeros(self.num, n_mat)
        for i, mat in enumerate(self.mat_lst):
            _, n0, k0 = get_index(mat, wlmin=self.wlmin, wlmax=self.wlmax, num=self.num, plot=False)  # get ind for i-th layer, [num]
            mat_n0[:, i] = n0
            mat_k0[:, i] = k0
        mat_k0[:, -1] = 0  # force the last layer to be lossless
        return mat_n0, mat_k0  # [num, n_mat)]

    def cal_trans(self, x, plot=False):
        """
        calculate the transmittance of the given multilayer, the 0-th layer is air by default with n = 1
        :param x: a list of thickness parameters, without the first air layer
        :return:
        """
        # notice air layer and last layer are not included in x, so x[i] is the thickness of i+1-th layer
        device = x.device.type
        n_mat = len(self.mat_lst)
        assert len(x) == n_mat - 2, 'number of thickness does not match the materials'
        # the initial transfer matrix   [num, 2, 2]
        smat = torch.eye(2).unsqueeze(0)
        smat = smat.repeat(self.num, 1, 1).type(torch.complex128).to(device)

        # iterate to calculate Iij * Lj from I01 * L1 to In-3,n-2 * Ln-2 and In-2,n-1
        for i in range(n_mat - 1):
            j = i + 1
            ni = torch.complex(self.mat_n0[:, i], self.mat_k0[:, i]).to(device)  # i-th layer index, [num]
            nj = torch.complex(self.mat_n0[:, j], self.mat_k0[:, j]).to(device)  # j-th layer index, [num]
            rij = (ni - nj) / (ni + nj)  # reflection coef
            tij = 2 * ni / (ni + nj)  # transmission coef
            # the Iij and Lj matrices are both with shape [num, 2, 2]
            Iij = torch.stack((torch.stack((torch.ones(self.num).to(device), rij), dim=1),
                               torch.stack((rij, torch.ones(self.num).to(device)), dim=1)), dim=2) \
                  / tij.unsqueeze(1).unsqueeze(1).type(torch.complex128)
            smat = smat.matmul(Iij)
            if i < n_mat - 2:
                Lj = torch.stack((torch.stack((torch.exp(-1j * 2 * np.pi * nj * x[i] / self.wl.to(device)),
                                               torch.zeros(self.num).to(device)), dim=1),
                                  torch.stack((torch.zeros(self.num).to(device),
                                               torch.exp(1j * 2 * np.pi * nj * x[i] / self.wl.to(device))),
                                              dim=1)), dim=2).type(torch.complex128)
                smat = smat.matmul(Lj)
        # extract refl and trans coefficient and intensity
        r = smat[:, 1, 0] / smat[:, 0, 0]
        t = 1 / smat[:, 0, 0]
        R = r.abs() ** 2
        T = t.abs() ** 2 * self.mat_n0[:, -1].to(device)
        if plot:
            T = T.cpu()
            plt.plot(self.wl, T, label='transmittance')
            # plt.plot(self.wl, R, label='reflectance')
            plt.plot(self.wl, self.target, label='target')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('trans/refl')
            plt.legend()
            plt.show()
        return T, R, t, r

    def cal_trans_batch(self, x, plot=False):
        """
        calculate the transmittance of a batch of given multilayer, the 0-th layer is air by default with n = 1
        :param x: a list of thickness parameters, without the first air layer and last layer [-1, len(mat_lst) - 2]
        :return:
        """
        # notice air layer and last layer are not included in x, so x[i] is the thickness of i+1-th layer
        bs = len(x)  # batch size
        device = x.device.type
        n_mat = len(self.mat_lst)
        assert len(x[0]) == n_mat - 2, 'number of thickness does not match the materials, one is {} the other is {}'.format(len(x[0]), n_mat-2)
        wl = self.wl.unsqueeze(1).repeat(1, bs).to(device)
        # the initial transfer matrix   [num, 2, 2]
        smat = torch.eye(2).unsqueeze(0).unsqueeze(0)
        smat = smat.repeat(bs, self.num, 1, 1).type(torch.complex128).to(device)

        # iterate to calculate Iij * Lj from I01 * L1 to In-3,n-2 * Ln-2 and In-2,n-1
        for i in range(n_mat - 1):
            j = i + 1
            ni = torch.complex(self.mat_n0[:, i], self.mat_k0[:, i]).to(device)  # i-th layer index, [num]
            nj = torch.complex(self.mat_n0[:, j], self.mat_k0[:, j]).to(device)  # j-th layer index, [num]
            rij = (ni - nj) / (ni + nj)  # reflection coef
            tij = 2 * ni / (ni + nj)  # transmission coef
            # the Iij and Lj matrices are both with shape [bs, num, 2, 2]
            Iij = torch.stack((torch.stack((torch.ones(self.num).to(device), rij), dim=1),
                               torch.stack((rij, torch.ones(self.num).to(device)), dim=1)), dim=2) \
                  / tij.unsqueeze(1).unsqueeze(1).type(torch.complex128)
            Iij = Iij.unsqueeze(0).repeat(bs, 1, 1, 1)
            smat = smat.matmul(Iij)
            nj = nj.unsqueeze(1).repeat(1, bs)
            if i < n_mat - 2:
                Lj = torch.stack((torch.stack((torch.exp(-1j * 2 * np.pi * nj * x[:, i] / wl).transpose(0, 1),
                                               torch.zeros(bs, self.num).to(device)), dim=2),
                                  torch.stack((torch.zeros(bs, self.num).to(device),
                                               torch.exp(1j * 2 * np.pi * nj * x[:, i] / wl).transpose(0, 1)),
                                              dim=2)), dim=3).type(torch.complex128)
                smat = smat.matmul(Lj)
        # extract refl and trans coefficient and intensity
        r = smat[:, :, 1, 0] / smat[:, :, 0, 0]
        t = 1 / smat[:, :, 0, 0]
        R = r.abs() ** 2
        T = t.abs() ** 2 * self.mat_n0[:, -1].to(device)  # [bs, num]
        if plot:
            plt.plot(self.wl, T[0].cpu(), label='transmittance')
            # plt.plot(self.wl, R, label='reflectance')
            plt.plot(self.wl, self.target, label='target')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('trans/refl')
            plt.legend()
            plt.show()
        return T, R, t, r

    def plot_config(self, x, save=None, tmin=None, tmax=None):
        """
        this function plot the design configuration with the layer thickness list x
        :param x: input thickness list
        :return: plot a figure of the structure
        """
        a = 0.8  # color alpha
        mat_lst = self.mat_lst[1:]  # ignore air layer
        mat_lst.reverse()
        # color for each material
        cmap = {'TiO2': [0.6, 0.6, 0.6, a], 'MgF2': [0.7, 0.6, 0.55, a],
                'Ag': [0.72, 0.75, 0.53, a], 'SiO2': [0.57, 0.66, 0.71, a],
                'HfO2': [0, 0.56, 0.45, a], 'SiC': [0.84, 0.41, 0, a],
                'ITO': [0.98, 0.23, 0.18, a], 'Al2O3': [0.46, 0.17, 0.53, a],
                'SiN': [0.14, 0.22, 0.29, a], 'Si': [0.2, 0.2, 0.2, a], 'air': [1, 1, 1, 1]}
        left = 1  # x_left
        h = 0  # initial height, change when adding material layer
        w = 4  # width of each layer
        hsub = 2  # height of substrate
        ratio = 100  # scale
        xf = x.flip(0)
        if tmin is not None:
            tminf = tmin.flip(0)
            tmaxf = tmax.flip(0)
        t = xf / ratio  # thickness of each layer in the plot
        fig, ax = plt.subplots(dpi=300)
        patches = []
        # add material caption
        for i, key in enumerate(cmap.keys()):
            cir = Circle((6, 6 - i), radius=0.3, facecolor=cmap[key])
            ax.text(6.5, 5.9 - i, key)
            patches.append(cir)
        # add substrate
        sub = Rectangle((left, h), w, hsub, facecolor=cmap[mat_lst[0]])
        h += hsub
        patches.append(sub)
        # add other layers
        for i in range(1, len(mat_lst)):
            # print(i, mat_lst[i], t[i-1].item(), h)
            rect = Rectangle((left, h), w, t[i - 1], facecolor=cmap[mat_lst[i]])
            # add text
            color = 'k'
            if tmin is not None:
                tmaxf = tmaxf.cpu()
                tminf = tminf.cpu()
                if (xf[i - 1] + 5 < tmaxf[i - 1] and xf[i - 1] - 5 > tminf[i - 1]):
                    color = 'k'
                elif xf[i - 1] + 5 > tmaxf[i - 1]:
                    color = 'r'
                else:
                    color = 'b'
            ax.text(-3, h + t[i - 1] / 2 - 0.1, '{:.1f} nm'.format(xf[i - 1].item()), color=color)
            h = h + t[i - 1]
            patches.append(rect)
        # plot all shapes
        for shape in patches:
            ax.add_patch(shape)
        plt.box('off')
        plt.axis('off')
        plt.axis('equal')
        if save is not None:
            plt.savefig(save)
        plt.show()

    def forward(self, x):
        if len(x.size()) == 1:
            spec, _, _, _ = self.cal_trans(x, plot=False)
        else:
            spec, _, _, _ = self.cal_trans_batch(x)
        return spec

    def generate_data(self, n, t_min=50, t_max=200):
        """
        generate training data
        :param t_max: maximum thickness
        :param t_min: minimum thickness
        :param n: data number
        :return: [layer, spec]  layer is [n, mat_lst - 2] thickness data, spec is [n, num] spec data
        """
        mat_lst = self.mat_lst[1:-1]
        n_mat = len(mat_lst)
        layer = torch.zeros(n, n_mat)
        for i in range(len(mat_lst)):
            if mat_lst[i] == 'Ag':
                layer[:, i] = torch.randint(15, 30, (n,))
            else:
                layer[:, i] = torch.randint(t_min, t_max, (n,))
        spec = self.forward(layer)
        return layer, spec

    def loss_function(self, spec):
        device = spec.device.type
        err = torch.mean(self.solar.to(device) * torch.abs(spec - self.target.to(device))) * self.num
        return err


if __name__ == '__main__':
    tm = transfer_matrix(mat_lst=['air', 'SiC', 'SiO2', 'Ag', 'MgF2', 'SiC', 'MgF2', 'TiO2', 'HfO2', 'Al2O3', 'SiC', 'ITO', 'TiO2', 'air'], num=500)
    x0 = torch.tensor([101, 152, 19, 146, 108, 102, 72, 0, 94, 144, 28, 93])
    # x0 = torch.tensor([91.1, 51.6, 190.3, 19.8, 154.1, 109.3])
    T, _, _, _ = tm.cal_trans(x0)
    mse0 = torch.mean((T - tm.target) ** 2)
    print('mse0 = {}'.format(mse0))
    # x = x.unsqueeze(0).repeat(256, 1)
    tm.plot_config(x0)
    T, R, t, r = tm.cal_trans(x0, plot=True)
    # t = time.time()
    # T = tm(x)
    # print(time.time()-t)
    # t = time.time()
    # for i in range(100):
    #     layer, spec = tm.generate_data(10000)
    #     print(time.time() - t, 'seconds')
    #     mse = tm.loss_function(spec)
    #     idx = torch.argmin(mse)
    #     print('min mse = {}'.format(mse[idx]))
    #     tm.plot_config(layer[idx])
    #     plt.plot(tm.wl, T, label='optim design')
    #     plt.plot(tm.wl, spec[0], label='optim design')
    #     plt.plot(tm.wl, tm.target, label='target')
    #     plt.xlabel('wavelength (nm)')
    #     plt.ylabel('trans/refl')
    #     plt.title('pred err {:.5f}, optim err {:.5f}'.format(mse[idx], mse0))
    #     plt.legend()
    #     plt.show()
