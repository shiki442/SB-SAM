import torch
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        # state['ema'].load_state_dict(loaded_state['ema'])
        state['ema'] = loaded_state['ema']
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state, epoch):
    step_now = epoch + state['step']
    path_ckpt = os.path.join(ckpt_dir, f'checkpoint_{step_now}.pth')
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        # 'ema': state['ema'].state_dict(),
        'ema': state['ema'],
        'step': epoch + state['step']
    }
    torch.save(saved_state, path_ckpt)
