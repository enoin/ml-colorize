import matplotlib.pyplot as plt
from nn.image_state_model import ImageProperties


def show_grid(train, eval):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(4):
        axs[0, i].imshow(train[i].permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        axs[0, i].axis('off')

        axs[1, i].imshow(eval[i, 0], cmap='gray')  # Grayscale image, no need to permute
        axs[1, i].axis('off')
    plt.show()


def show_test_images(images: list[ImageProperties]):
    fig, axs = plt.subplots(3, images.__len__(), figsize=(12, 6))
    plt.subplots_adjust(wspace=0.012, hspace=0.001)

    for i, image in enumerate(images):
        axs[0, i].imshow(image.gray.permute(1, 2, 0), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_aspect('equal')
        axs[1, i].set_aspect('equal')
        axs[2, i].set_aspect('equal')

        axs[1, i].imshow(image.colorized.permute(1, 2, 0))
        axs[1, i].axis('off')
        axs[1, i].set_ylabel('sdc')

        axs[2, i].imshow(image.ground_truth.permute(1, 2, 0))
        axs[2, i].axis('off')
    plt.show()
