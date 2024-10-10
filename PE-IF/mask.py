import cv2
import numpy as np

def get_mask(img, color_lb, color_ub, connected_region=False):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lb = np.array(color_lb)
    ub = np.array(color_ub)
    mask = cv2.inRange(img, lb, ub)
    if mask.sum() == 0:
        return mask
    else:
    # select the maximum connected region as the mask
        if connected_region:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype('uint8')
        return mask

def mask_image(vrep_rgb_pth = "cam2.png", nerf_rgb_pth = "nerf_cam2.png", vrep_mask_outdir="./outdir/vrep_mask.png", masked_nerf_outdir="./outdir/masked_vrep_rgb.png",lb = np.array([0,200,0]), ub = np.array([0,255,0])):
    # read in the two images
    vrep_rgb = cv2.imread(vrep_rgb_pth)

    # import ipdb; ipdb.set_trace()
    print(vrep_rgb_pth)
    vrep_mask = get_mask(vrep_rgb, lb, ub)
    cv2.imwrite(vrep_mask_outdir, vrep_mask)
    nerf_rgb = cv2.imread(nerf_rgb_pth)
    if not nerf_rgb is None:
        import ipdb; ipdb.set_trace()
        masked_vrep_rgb = vrep_rgb.copy()
        masked_vrep_rgb[np.where(vrep_mask == 255)] = nerf_rgb[np.where(vrep_mask == 255)]
        print(cv2.imwrite(masked_nerf_outdir, masked_vrep_rgb))
    else:
        print("only generate mask, no nerf_rgb to mask")
    

if __name__ == "__main__":
    for i in range(10):
        mask_image(vrep_rgb_pth = f"./vrep_img/vrep_{i}.png", nerf_rgb_pth = f"./nerf_img/{i}_image1.png", masked_nerf_outdir=f"./outdir/masked_img/{i}.png")