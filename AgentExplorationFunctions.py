import torch
import matplotlib.pyplot as plt


def agent_reached_goal(agent, environment, goal_index):
    target_goal_layer = 0 if goal_index == environment.object_type_num else goal_index.item() + 1
    same_elements_on_grid_num = torch.logical_and(environment.env_map[0, 0, :, :],
                                                  environment.env_map[0, target_goal_layer, :, :]).sum()
    if same_elements_on_grid_num > 0:
        return True
    return False


def update_pre_located_objects(object_locations, agent_location, goal_reached, goal_type):
    pre_located_objects = []
    if goal_reached:
        for obj_type in object_locations:
            temp = []
            for loc in obj_type:
                if any(~torch.eq(loc, agent_location[0])):
                    temp.append(loc.tolist())
                else:
                    temp.append([])
            pre_located_objects.append(temp)
    return pre_located_objects


def get_controller_loss(controller_at_loss, device):
    if torch.is_tensor(controller_at_loss):
        return controller_at_loss
    else:
        return torch.Tensor([0.0]).to(device)


def get_meta_controller_loss(meta_controller_at_loss):
    if torch.is_tensor(meta_controller_at_loss):
        return meta_controller_at_loss.detach().item()
    else:
        return 0.0
