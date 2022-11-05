from forward_model import transfer_matrix as tmm
from utils import *

if __name__ == '__main__':
    gamma = 0.99
    spec_dim = 500
    num = 500
    wlmin = 300
    wlmax = 2500
    tm = tmm(mat_lst=['air', 'TiO2', 'SiO2', 'SiC', 'SiO2', 'Ag', 'TiO2', 'SiO2', 'SiN', 'SiC', 'SiO2'],
             num=num, wlmin=wlmin, wlmax=wlmax)
    target = tm.target
    wl = tm.wl
    lst = []
    x0 = torch.tensor([88., 64, 114, 142, 17, 188, 84, 202, 6])
    lst.append(x0)
    x0 = torch.tensor([88., 67, 115, 140, 21, 188, 88, 151, 48])
    lst.append(x0)
    lab = ['GA optim', 'DL optim']
    plt.rcParams['font.size'] = 20
    plt.figure(dpi=300)
    ep = 2000
    for idx, x0 in enumerate(lst):
        lr = 1000
        print('idx = {}'.format(idx))
        x = torch.nn.parameter.Parameter(x0, requires_grad=True)
        err = torch.zeros(ep)
        for i in range(ep):
            y = tm(x)
            loss = tm.loss_function(y)
            err[i] = loss.item()
            grad = torch.autograd.grad(loss, x)[0]
            x = x - grad * lr
            lr = lr * gamma
            # print(x)
        plt.plot(torch.arange(ep), err, label=lab[idx])
        plt.xlabel('epoch')
        plt.ylabel('error')
        # plt.ylim(0, 0.023)
        print(x, err[-1])
    plt.legend()
    plt.show()
