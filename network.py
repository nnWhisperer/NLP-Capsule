import torch
import torch.nn as nn
from layer import PrimaryCaps, FCCaps, FlattenCaps


def BCE_loss(x, target):
    return nn.BCELoss()(x.squeeze(2), target)


class CapsNet_Text(nn.Module):
    def __init__(self, args, w2v):
        super(CapsNet_Text, self).__init__()
        self.num_classes = args.num_classes
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))

        self.ngram_size = [2, 4, 8]
        stride = 2
        self.convs_doc = nn.ModuleList([nn.Conv1d(args.agent_length + args.cust_length, 32, K, stride=stride) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=args.dim_capsule, in_channels=32, out_channels=32, kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()
        W_doc_dim = 32 * torch.sum(torch.tensor([(args.vec_size - i + 2) // stride for i in self.ngram_size]))

        self.W_doc = nn.Parameter(torch.FloatTensor(W_doc_dim, args.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(args, output_capsule_num=args.num_classes,
                                            input_capsule_num=args.num_compressed_capsule,
                                            in_channels=args.dim_capsule, out_channels=args.dim_capsule)

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def forward(self, data, labels):
        data = self.embed(data)
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](data)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, activations = self.fc_capsules_doc_child(poses, activations, labels)
        return poses, activations
