import numpy as np
import matplotlib.pyplot as plt

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def visualize_multiple_rows(*image_lists, names=None):
    n = len(image_lists)
    m = len(image_lists[0])

    fig, axes = plt.subplots(m, n, figsize=(12, 60))

    for i, image_list in enumerate(image_lists):
        for j, image in enumerate(image_list):
            axes[j, i].imshow(image)
            axes[j, i].axis("off")
    plt.show()


def visualize_random_rows(show_num, *image_lists, names=None):
    n = len(image_lists)
    m = show_num

    fig, axes = plt.subplots(m, n, figsize=(12, 60))

    for i, a in enumerate(np.random.randint(len(image_lists[0]), size=(m))):
        for j, image_list in enumerate(image_lists):
            axes[i, j].imshow(image_list[a])
            axes[i, j].axis("off")

    plt.show()