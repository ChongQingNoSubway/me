

import torch 

# lesion_norm  (num_les,embedding_dim)
div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1e-10*torch.eye(args.num_les).cuda())
