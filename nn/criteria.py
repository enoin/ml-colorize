from torch import nn
from torchvision import models


class ColorLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.5):
        super(ColorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_mse = lambda_mse

        vgg19 = models.vgg19(pretrained=True).features
        self.feature_extractor = vgg19.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lambda_perceptual = lambda_perceptual

    def forward(self, input_image, target_image):
        mse_loss = self.mse_loss(input_image, target_image)
        perceptual_loss = self.calculate_perceptual_loss(input_image, target_image)
        total_loss = self.lambda_mse * mse_loss + self.lambda_perceptual * perceptual_loss

        return total_loss

    def calculate_perceptual_loss(self, input_image, target_image):
        input_features = self.feature_extractor(input_image)
        target_features = self.feature_extractor(target_image)
        perceptual_loss = nn.functional.mse_loss(input_features, target_features)

        return perceptual_loss
