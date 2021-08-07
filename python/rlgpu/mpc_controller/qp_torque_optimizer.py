"""Set up the zeroth-order QP problem for stance leg control.

For details, please refer to section XX of this paper:
https://arxiv.org/abs/2009.10019
"""

import torch
from qpth.qp import QPFunction
from torch.autograd import Variable

ACC_WEIGHT = torch.as_tensor([1., 1., 1., 10., 10, 1.])


def compute_mass_matrix(num_envs, robot_mass, robot_inertia, foot_positions, device):
    rot_z = torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
    inv_mass = (torch.eye(3, device=device) / robot_mass).unsqueeze(0).repeat(num_envs, 1, 1)
    inv_inertia = torch.inverse(robot_inertia).unsqueeze(0).repeat(num_envs, 1, 1)
    mass_mat = torch.zeros((num_envs, 6, 12), device=device)
    for leg_id in range(4):
        mass_mat[:, :3, leg_id * 3:leg_id * 3 + 3] = inv_mass
        x = foot_positions[:, leg_id]
        foot_position_skew = torch.stack([torch.stack([0 * x[..., 0], -x[..., 2], x[..., 1]], dim=-1), torch.stack([x[..., 2], 0 * x[..., 0], -x[..., 0]], dim=-1),
                                          torch.stack([-x[..., 1], x[..., 0], 0 * x[..., 0]], dim=-1)], dim=-2)
        mass_mat[:, 3:6, leg_id * 3:leg_id * 3 +
                 3] = torch.bmm(torch.bmm(rot_z.transpose(-2, -1), inv_inertia), foot_position_skew)
    return mass_mat


def compute_constraint_matrix(mpc_body_mass,
                              contacts,
                              friction_coef=0.8,
                              f_min_ratio=0.1,
                              f_max_ratio=10,
                              device="cuda:0"
                              ):
    num_envs = contacts.shape[0]
    f_min = f_min_ratio * mpc_body_mass * 9.81
    f_max = f_max_ratio * mpc_body_mass * 9.81

    A = torch.zeros((24, 12), device=device).unsqueeze(0).repeat(num_envs, 1, 1)
    lb = torch.zeros((24), device=device).unsqueeze(0).repeat(num_envs, 1)
    for leg_id in range(4):
        A[:, leg_id * 2, leg_id * 3 + 2] = 1
        A[:, leg_id * 2 + 1, leg_id * 3 + 2] = -1
        contacts_indices = torch.nonzero(contacts[:, leg_id])
        non_contacts_indices = torch.nonzero(contacts[:, leg_id] != 0)
        if contacts_indices.shape[-1] > 0:
            lb[contacts_indices, leg_id * 2], lb[contacts_indices,
                                                 leg_id * 2 + 1] = f_min, -f_max
        if non_contacts_indices.shape[-1] > 0:
            lb[non_contacts_indices, leg_id] = -1e-7

    # Friction constraints
    for leg_id in range(4):
        row_id = 8 + leg_id * 4
        col_id = leg_id * 3
        lb[:, row_id:row_id + 4] = torch.as_tensor(
            [0, 0, 0, 0], device=device).unsqueeze(0).repeat(num_envs, 1)
        A[:, row_id, col_id:col_id +
            3] = torch.as_tensor([1, 0, friction_coef], device=device).unsqueeze(0).repeat(num_envs, 1)
        A[:, row_id + 1, col_id:col_id +
            3] = torch.as_tensor([-1, 0, friction_coef], device=device).unsqueeze(0).repeat(num_envs, 1)
        A[:, row_id + 2, col_id:col_id +
            3] = torch.as_tensor([0, 1, friction_coef], device=device).unsqueeze(0).repeat(num_envs, 1)
        A[:, row_id + 3, col_id:col_id +
            3] = torch.as_tensor([0, -1, friction_coef], device=device).unsqueeze(0).repeat(num_envs, 1)
    return A, lb


def compute_objective_matrix(mass_matrix, desired_acc, acc_weight, reg_weight, device):
    num_envs = mass_matrix.shape[0]
    g = torch.as_tensor([0., 0., 9.8, 0., 0., 0.],
                        device=device).repeat(num_envs, 1)
    Q = torch.diag(acc_weight).unsqueeze(0).repeat(num_envs, 1, 1)
    R = (torch.ones((12, 12), device=device) * reg_weight).repeat(num_envs, 1, 1)
    quad_term = torch.bmm(
        torch.bmm(mass_matrix.transpose(-2, -1), Q), mass_matrix) + R
    linear_term = torch.bmm(
        torch.bmm((g + desired_acc).unsqueeze(1), Q), mass_matrix)
    return quad_term, linear_term


def compute_contact_force(robot_task,
                          desired_acc,
                          contacts,
                          acc_weight=ACC_WEIGHT,
                          reg_weight=1e-4,
                          friction_coef=0.45,
                          f_min_ratio=0.1,
                          f_max_ratio=10.):
    device = robot_task.device
    num_envs = robot_task.num_envs
    mass_matrix = compute_mass_matrix(
        num_envs,
        108 / 9.81,
        torch.as_tensor([0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00],
                        device=device).view((3, 3)),
        robot_task._footPositionsInBaseFrame(), device)
    G, a = compute_objective_matrix(mass_matrix, desired_acc, acc_weight.to(device),
                                    reg_weight, device)
    C, b = compute_constraint_matrix(108 / 9.81, contacts,
                                     friction_coef, f_min_ratio, f_max_ratio, device)
    G += 1e-4 * torch.eye(12).unsqueeze(0).repeat(num_envs, 1, 1).to(device)
    e = Variable(torch.Tensor()).to(device)
    result = QPFunction(verbose=False)(G, -a.squeeze(1), -C, -b, e, e)
    return -result.view((robot_task.num_envs, 4, 3))
