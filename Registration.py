import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math
from scipy.interpolate import interpn
def find_match(template, target):

    sift = cv2.xfeatures2d.SIFT_create()
    keyp1, des1 = sift.detectAndCompute(template, None)
    keyp2, des2 = sift.detectAndCompute(target, None)

    knn = NearestNeighbors(n_neighbors=2).fit(des2)
    distance, indices = knn.kneighbors(des1)
    x1 = []
    x2 = []
    for i in range(len(distance)):
        if distance[i][0] < 0.70 * distance[i][1]:
            x1.append(keyp1[i].pt)
            x2.append(keyp2[indices[i][0]].pt)

    x1 = np.array(x1)
    x2 = np.array(x2)

    return x1,x2

def align_image_using_feature(template, target, ransac_thr, ransac_iter):

    max_inliers = 0
    A = np.zeros((6,6))

    for iter in range(ransac_iter):
        random_index = np.random.choice(template.shape[0], 3, replace=False)

        x1 = template[random_index[0]][0]
        y1 = template[random_index[0]][1]
        x2 = template[random_index[1]][0]
        y2 = template[random_index[1]][1]
        x3 = template[random_index[2]][0]
        y3 = template[random_index[2]][1]

        temp_A = np.array([[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1], [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1],
                           [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1]])
        target_point = target[random_index].reshape(-1)

        affine_coef = np.linalg.solve(temp_A, target_point)
        affine_coef = np.array(list(affine_coef)+[0,0,1]).reshape((3,3))

        inliers = 0
        for i in range(len(template)):
            cur_point = np.ones(3)
            cur_point[0] = template[i][0]
            cur_point[1] = template[i][1]

            transform_point = np.ones(3)
            transform_point[0] = target[i][0]
            transform_point[1] = target[i][1]

            estimate = np.dot(affine_coef, cur_point.reshape(-1))
            diff = np.sqrt(np.sum((estimate - transform_point) ** 2))

            if diff < ransac_thr:
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            A = affine_coef

    return A

def warp_image(img, A, output_size):

    x = [i for i in range(len(img))]
    y = [i for i in range(len(img[0]))]
    points = (x,y)
    row, col = output_size
    img_warped = np.zeros(output_size)
    for i in range(row):
        print("warping",i)
        for j in range(col):

            template_point = np.ones(3)
            template_point[0] = j
            template_point[1] = i

            target_point = np.dot(A,template_point.reshape(-1))
            temp = np.zeros(2)
            temp[0] = target_point[1]
            temp[1] = target_point[0]

            template_value = interpn(points, img, temp)
            img_warped[i][j] = template_value


    return img_warped


def filter_image(im, filter):

    surroundedzero = np.pad(im, 1)
    im_filtered = np.zeros((im.shape[0], im.shape[1]))
    direction = [-1, 0, 1]
    for row in range(1, len(surroundedzero) - 1):
        for col in range(1, len(surroundedzero[0]) - 1):
            current_sum = 0
            for i in direction:
                for j in direction:
                    current_sum += surroundedzero[row + i][col + j] * filter[i + 1][j + 1]
            im_filtered[row - 1][col - 1] = current_sum

    return im_filtered

def get_gradient(grad_x,grad_y):
    row, col = grad_x.shape[0], grad_y.shape[1]
    gradient = np.zeros((row,col,2))
    for i in range(row):
        for j in range(col):
            gradient[i][j] = [grad_x[i][j], grad_y[i][j]]

    return gradient

def align_image(template, target, A):
    sobel_filter_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    sobel_filter_y = np.transpose(sobel_filter_x)
    # sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobel_filter_y = np.transpose(sobel_filter_x)
    grad_x = filter_image(template, sobel_filter_x)
    grad_y = filter_image(template, sobel_filter_y)
    gradient = get_gradient(grad_x, grad_y)

    row, col = len(template),len(template[0])
    hessian = np.zeros((6,6))
    steepest = np.zeros((row,col,6))
    for i in range(row):
        for j in range(col):
            Jacobian = np.array([[j,i,1,0,0,0],[0,0,0,j,i,1]])
            cur_steepest = np.dot(gradient[i][j],Jacobian)
            steepest[i][j] = cur_steepest
            hessian += np.dot(np.transpose(cur_steepest).reshape(6, 1), cur_steepest.reshape(1, 6))

    hessian_inverse = np.linalg.inv(hessian)
    error = float("inf")
    iter = 0
    errorlist = []
    A_refined = A
    while error > 2e3 and iter < 250:
        warp_target = warp_image(target,A_refined, template.shape)
        Ierror = warp_target - template
        error = np.sqrt(np.sum((Ierror) ** 2))
        errorlist.append(error)
        F = np.zeros((6, 1))
        for i in range(row):
            for j in range(col):
                F += np.dot(np.transpose(steepest[i][j]), Ierror[i][j]).reshape(6,1)

        delta_p = np.dot(hessian_inverse, F)
        affine_delta_p = np.array([[delta_p[0][0]+1,delta_p[1][0], delta_p[2][0]],
                                   [delta_p[3][0], delta_p[4][0]+1, delta_p[5][0]],
                                   [0,0,1]])
        A_refined = np.dot(A_refined,np.linalg.inv(affine_delta_p))
        iter += 1
        print(iter,error)

    return A_refined,errorlist


def track_multi_frames(template, img_list):
    # To do
    x1, x2 = find_match(template, img_list[0])
    ransac_thr, ransac_iter = 9, 500
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    A_list = []

    for i in range(len(img_list)):
        A, errors = align_image(template, img_list[i], A)
        A_list.append(A)
        template = warp_image(img_list[i], A, template.shape)

    return A_list

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()

def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr, ransac_iter = 9 ,500
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    errors = np.array(errors)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


