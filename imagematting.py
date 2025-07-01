import cv2
import numpy as np

def estimate_alpha(image, trimap):
    
    image = image.astype(np.float32) / 255.0

    
    trimap = trimap.astype(np.float32) / 255.0

    
    foreground = np.where(trimap > 0.95, 1.0, 0.0)  
    alpha = np.where(trimap > 0.05, 1.0, 0.0)  
    for _ in range(5):  
        alpha = (image[:, :, 0] - image[:, :, 2] * alpha) / (1e-12 + foreground + (1.0 - trimap) * alpha)
        alpha = np.clip(alpha, 0, 1)

    return alpha


if __name__ == "__main__":
    
    image = cv2.imread("R:\MSc IT\sem 2\computer vision\model.jpg")
    trimap = cv2.imread("R:\MSc IT\sem 2\computer vision\model.jpg", cv2.IMREAD_GRAYSCALE)

   
    alpha = estimate_alpha(image, trimap)

    
    cv2.imshow("Alpha Matte", alpha)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
