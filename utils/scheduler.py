import torch

def get_scheduler(optimizer, hyps, verbose = False):

    lr_scheduler = hyps['lr_scheduler']

    match lr_scheduler:
        case 'ReduceLROnPlateau':
            # patience: Number of epochs with no improvement after which learning rate will be reduced.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=hyps['better'],
                                    factor=0.1, patience=5,
                                    threshold=0.0001, threshold_mode='rel', 
                                    cooldown=0, min_lr=float(hyps['min_lr']), eps=1e-08, verbose=verbose)
            
        case 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=float(hyps['min_lr']), 
                                    last_epoch=- 1, verbose=verbose)

        case 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(hyps['max_lr']), total_steps=10, epochs=None, 
                                    steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', 
                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, 
                                    div_factor=25.0, final_div_factor=10000.0, three_phase=False, 
                                    last_epoch=- 1, verbose=verbose)
        case _:
            raise ValueError("Undefined scheduler")
    
    return scheduler
