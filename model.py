import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class ResNet18Attention(nn.Module):

    def __init__(self):
        super(ResNet18Attention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # model = resnet18()
        # num_input_channel = 1
        # model.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=3, stride=1, bias=False)
        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        # self.backbone = create_feature_extractor(
        #    model, return_nodes={'layer4': 'extracted_features'})
        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        # self.pooling = nn.Sequential(nn.Linear(512 * 1 * 1, self.L),
        #                             nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1)
            ## nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = torch.squeeze(x, 0)

        H = self.backbone(x)  # ["extracted_features"]
        # print("backbone max and min, mean", torch.max(H), torch.min(H), torch.mean(H))
        # print("backbone shape:")
        # print(H.shape)
        # print(torch.isnan(H).any())
        H = self.adaptive_pooling(H)
        # print("adaptive pooling max and min", torch.max(H), torch.min(H))
        H = H.view(-1, 512 * 1 * 1)
        # print("pooling output:", H)
        # print(H.shape)
        # print(torch.isnan(H).any())
        # H = self.pooling(H)
        # print("pooling2 output:", H)
        # print(H.shape)
        # print(torch.isnan(H).any())
        A = self.attention(H)  # NxK
        # print("attention output:", A)
        # print(A.shape)
        A = torch.transpose(A, 1, 0)  # KxN
        # print("shape after transposing ", A.shape)
        A = F.softmax(A, dim=1)  # softmax over N
        # print("attention after softmax output:", A)
        # print(A.shape)
        M = torch.mm(A, H)  # KxL
        # print("after applying attention:", M)
        # print(M.shape)
        Y_prob = self.classifier(M)

        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        # _, Y_hat, _ = self.forward(X)
        # print("error calc: Y_hat, Y, eq():", Y_hat, Y, Y_hat.eq(Y))
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def calculate_objective(self, Y, Y_prob):
        Y = Y.float()
        # Y_prob, _, A = self.forward(X)
        print("calc objective Y,Y_prob: ")
        print(Y, Y_prob)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (
                Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        print("nll: ", neg_log_likelihood)
        print(neg_log_likelihood.data)
        return neg_log_likelihood


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
