import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.utils import adjust_learning_rate, optim
from models.vicreg import VICRegNet
from utils.loss import sim_loss, cov_loss, std_loss
from utils.dataset import TimeSeriesDataset
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VICReg Training')
parser.add_argument('--path', help='path to datasets')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
parser.add_argument('--device', default='cuda', type=str, help='device for training')
parser.add_argument('--l', default=25,  help='coefficients of the invariance')
parser.add_argument('--mu', default=25,  help='coefficients of the variance')
parser.add_argument('--nu', default=1,  help='coefficients of the covariance')
parser.add_argument('--weight_decay', default=1e-6,  help='weight decay')
parser.add_argument('--lr', default=0.2,  help='weight decay')
parser.add_argument('--epoch', default=100,  help='number of epochs')
parser.add_argument('--log_dir', default=r'logs', help = 'directory to save logs')
parser.add_argument('--save_chpt', default = 'checkpoints', help = 'path to save checkpoints')
parser.add_argument('--save_freq', default=1000, help='step frequency to save checkpoints')

def main():
    print('Training Starts')
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.log_dir)

    # load dataset
    dataset = TimeSeriesDataset(path=args.path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,drop_last=True, shuffle=False,num_workers=args.num_workers, prefetch_factor=2) 

    # create model
    model = VICRegNet().to(args.device)
    optimizer = optim(model, args.weight_decay)

    os.makedirs(os.path.join('.', args.save_chpt), exist_ok=True)

    for epoch in range(0, args.epoch):
        loop = tqdm(enumerate(dataloader, start=epoch * len(dataloader)), total=len(dataloader), leave=False)
        for step, ((view_a, view_b)) in loop:
            adjust_learning_rate(args, optimizer, dataloader, step)
            optimizer.zero_grad()
            view_a = view_a.to(args.device, dtype=torch.float)
            view_b = view_b.to(args.device, dtype=torch.float)
            repr_a, repr_b = model(view_a, view_b)

            _sim_loss = sim_loss(repr_a, repr_b)
            _std_loss = std_loss(repr_a, repr_b)
            _cov_loss = cov_loss(repr_a, repr_b)

            loss = args.l * _sim_loss + args.mu * _std_loss + args.nu * _cov_loss
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Loss/sim", _sim_loss, epoch)
            writer.add_scalar("Loss/std", _std_loss, epoch)
            writer.add_scalar("Loss/cov", _cov_loss, epoch)

            if step % int(args.save_freq) == 0 and step:
                with open(os.path.join(args.log_dir, 'logs.txt'), 'a') as log_file:
                    log_file.write(f'Epoch: {epoch}, Step: {step}, Train loss: {loss.cpu().detach().numpy()}, Sim loss: {_sim_loss.cpu().detach().numpy()}, Std loss: {_std_loss.cpu().detach().numpy()}, Cov loss: {_cov_loss.cpu().detach().numpy()}\n')


            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')
            loop.set_postfix(loss = loss.cpu().detach().numpy())

        print(f'Loss for epoch {epoch} is {loss.cpu().detach().numpy()}')
    print('End of the Training. Saving final checkpoints.')
    state = dict(epoch=args.epoch, model=model.state_dict(),
                 optimizer=optimizer.state_dict())
    torch.save(state, os.path.join('.', args.save_chpt,  'final_checkpoint.pth'))
    writer.flush()
    writer.close()

if '__main__' == __name__:
    main()
    

