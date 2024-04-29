import torch
import numpy as np
from torch.autograd import Variable


def calculate_immediate_sensitivity(model, criterion, inputs, labels):
    #inp = Variable(inputs, requires_grad=True)
    inp = torch.tensor(inputs, requires_grad=True)

    outputs = model.forward(inp)
    loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))

    # (1) first-order gradient (wrt parameters)
    first_order_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    # (2) L2 norm of the gradient from (1)
    grad_l2_norm = torch.norm(torch.cat([x.view(-1) for x in first_order_grads]), p=2)

    # (3) Gradient (wrt inputs) of the L2 norm of the gradient from (2)
    sensitivity_vec = torch.autograd.grad(grad_l2_norm, inp, retain_graph=True)[0]

    # (4) L2 norm of (3) - "immediate sensitivity"
    s = [torch.norm(v, p=2).cpu().numpy().item() for v in sensitivity_vec]

    loss.backward()
    return loss, s, outputs

