from tqdm import tqdm
from tensorboardX import SummaryWriter
from typing import Sequence, Iterable
import torch

def train(model, data_loader, optimizer, loss_fn, epoch, device,\
          event_writer: SummaryWriter=None, scheduler=None,\
          log_interval=100, use_tqdm=False):
    model.train()
    model.to(device)
    if use_tqdm: data_loader = tqdm(data_loader)

    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % log_interval == 0:
            if event_writer:
                stp = epoch*len(data_loader)+batch_idx
                event_writer.add_scalar('loss', loss, stp)
                event_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], stp)
            if use_tqdm: data_loader.set_description(
                f'Train epoch {epoch} Loss: {loss.item():.6f}')

    if scheduler: scheduler.step()
    return train_loss / len(data_loader)


def test(model, data_loader, loss_fn, epoch, device, \
         event_writer: SummaryWriter = None, test_type='test'):
    model.eval()
    model.to(device)
    test_loss, correct, tot = 0, 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            tot += len(target)

    test_loss /= len(data_loader)
    acc = correct / tot
    if event_writer:
        event_writer.add_scalar(test_type+'/loss', test_loss, epoch)
        event_writer.add_scalar(test_type+'/acc', acc, epoch)

    return test_loss, acc


def train_with_recorders(model, data_loader, optimizer, loss_fn, epoch, \
                         device, recorders=None, \
                         event_writer: SummaryWriter=None, \
                         scheduler=None, log_interval=100, use_tqdm=False):
    model.train()
    model.to(device)
    if use_tqdm: data_loader = tqdm(data_loader)

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if recorders and isinstance(recorders, Iterable):
            for rcd in recorders:
                rcd.record(len(target), output, target, loss, optimizer)

    if scheduler: scheduler.step()
    return 

def test_with_recorders(model, data_loader, loss_fn, epoch, device, \
                        recorders=None):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target) # sum up batch loss
            if recorders and isinstance(recorders, Iterable):
                for rcd in recorders:
                    rcd.record(len(target), output, target, loss)
    
    return
