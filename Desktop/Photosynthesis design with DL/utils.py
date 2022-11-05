from scipy.interpolate import interp1d
import errno
import numpy as np
import os
import sys
import torch
import time
import matplotlib.pyplot as plt
from scipy.io import *


def get_index(mat, wlmin=300, wlmax=2500, num=1000, plot=False):
    """
    get the material index
    :param mat: str of material name, choice=[Ag, MgF2, SiO2, TiO2]
    :param wlmin: min wavelength, unit: nm
    :param wlmax: max wavelength, unit: nm
    :param num: data point number
    :return: torch.tensor of the wavelength and complex refractive index, from wlmin to wlmax with num equal-dist points
    """
    wl = np.linspace(wlmin, wlmax, num)
    if mat == 'air':
        n0 = np.ones_like(wl)
        k0 = np.zeros_like(wl)
    else:
        data = np.loadtxt(os.path.join('material_nk - 0529', '{}.txt'.format(mat)))
        fn = interp1d(data[:, 0], data[:, 1])
        fk = interp1d(data[:, 0], data[:, 2])
        n0 = fn(wl)
        k0 = fk(wl)
    if plot:
        plt.plot(wl, n0, label='n')
        plt.plot(wl, k0, label='k')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('index value')
        plt.legend()
        plt.title('refractive index of {}'.format(mat))
        plt.show()
    return torch.from_numpy(wl), torch.from_numpy(n0), torch.from_numpy(k0)


def target_trans(wlmin=300, wlmax=2500, num=1000, plot=False):
    """
    get the target transmissivity spectrum
    :param wlmin: min wavelength, unit: nm
    :param wlmax: max wavelength, unit: nm
    :param num: data point number
    :return: np.array of the target transmission, from wlmin to wlmax with num equal-dist points
    """
    wl = np.linspace(wlmin, wlmax, num)
    spe = np.logical_or(np.logical_and(wl > 400, wl < 500), np.logical_and(wl > 600, wl < 700))
    spe = spe.astype(np.float)
    if plot:
        plt.plot(wl, spe, label='n')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('transmissivity')
        plt.legend()
        plt.title('target transmission spectrum')
        plt.show()
    return torch.from_numpy(spe)

def solar(wlmin=300, wlmax=2500, num=1000, plot=False):
    """
    get the target transmissivity spectrum of solar
    :param wlmin: min wavelength, unit: nm
    :param wlmax: max wavelength, unit: nm
    :param num: data point number
    :return: np.array of the target transmission, from wlmin to wlmax with num equal-dist points
    """
    spe = loadmat('material_nk - 0529/AM1.5')
    wl0 = spe['lambda_solar'][:, 0] * 1000    # wavelength nm
    trans0 = spe['spectrum_solar'][:, 0]  # transmittance
    wl = np.linspace(wlmin, wlmax, num)
    fn = interp1d(wl0, trans0)
    spe = fn(wl)
    spe = spe / np.sum(spe)
    if plot:
        plt.plot(wl, spe, label='n')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('transmissivity')
        plt.legend()
        plt.title('solar spectrum')
        plt.show()
    return torch.from_numpy(spe)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def write_record(file_path, str):
    if not os.path.exists(file_path):
        # os.makedirs(file_path)
        os.system(r"touch {}".format(file_path))
    f = open(file_path, 'a')
    f.write(str)
    f.close()


def count_parameters(model, all=True):
    # If all= Flase, we only return the trainable parameters; tested
    return sum(p.numel() for p in model.parameters() if p.requires_grad or all)


def adjust_learning_rate(optimizer, epoch, lr, factor=0.1, step=30):
    """Sets the learning rate to the initial LR decayed by factor every step epochs"""
    lr = lr * (factor ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_model(net, optimizer, epoch, path, **kwargs):
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, path)


if __name__ == '__main__':
    solar(plot=True)