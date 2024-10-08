from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import lpips
def calc_psnr(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).
        
    References
    -------
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    '''
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img2 = img2.resize(img1.size)
    # import ipdb; ipdb.set_trace()
    img1, img2 = np.array(img1)[...,:3], np.array(img2)[...,:3]
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


# coding:utf-8



def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.
        
    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score

# coding:utf-8



class util_of_lpips():
    def __init__(self, net="vgg", use_gpu=False):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01
    
lpips_fn = util_of_lpips()
for obj in ["scene", "fixer"]:
    print("PSNR =", calc_psnr("{}_gt.png".format(obj), "{}_render.png".format(obj)))
    print("SSIM =", calc_ssim("{}_gt.png".format(obj), "{}_render.png".format(obj)))
    print("LPIPS =", lpips_fn.calc_lpips("{}_gt.png".format(obj), "{}_render.png".format(obj)).item())

# concat image vertically in cv2
import cv2
img = []
for i in range(4):
    g = cv2.imread("scene_render_ngp{}.png".format(i+1))
    img.append(g)

img = cv2.vconcat(img)
img = cv2.resize(img, (642, 1442))
cv2.imwrite("scene_render_ngp_concat.png", img)

print("PSNR =", calc_psnr("scene_gt.png", "scene_render_ngp_concat.png"))
print("SSIM =", calc_ssim("scene_gt.png", "scene_render_ngp_concat.png"))
print("LPIPS =", lpips_fn.calc_lpips("scene_gt.png", "scene_render_ngp_concat.png").item())
