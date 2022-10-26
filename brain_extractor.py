import SimpleITK as sitk
import numpy as np
import cv2


class BrainExtractor(object):
    def __int__(self):
        pass

    def array2image(self, array, orig_img, new_spacing=None):
        rec_img = sitk.GetImageFromArray(array)
        if new_spacing is not None:
            rec_img.SetSpacing(new_spacing)
        rec_img.SetDirection(orig_img.GetDirection())
        rec_img.SetOrigin(orig_img.GetOrigin())
        rec_img.SetSpacing(orig_img.GetSpacing())
        return rec_img

    def binarize_img(self, img, itk_img, th):
        binary_img = np.zeros_like(img).astype(np.int)
        binary_img[img >= th] = 255
        binary_img[img < th] = 0
        return self.array2image(binary_img, itk_img)

    def _fill_hole(self, mask):
        contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for i in range(len(contours)):
            drawing = np.zeros_like(mask, np.uint8)
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)
        res = sum(contour_list)
        return np.squeeze(res)

    def _get_largest_connected_component(self, itk_binary_img):
        cc = sitk.ConnectedComponent(itk_binary_img)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetGlobalDefaultNumberOfThreads(8)
        stats.Execute(cc, itk_binary_img)
        max_label = 0
        max_size = 0
        for l in stats.GetLabels():
            size = stats.GetPhysicalSize(l)
            if max_size < size:
                max_label = l
                max_size = size
        label_mask_img = sitk.GetArrayFromImage(cc)
        res_mask = label_mask_img.copy()
        res_mask[label_mask_img == max_label] = 255
        res_mask[label_mask_img != max_label] = 0
        itk_res_mask = self.array2image(res_mask)
        return itk_res_mask

    def get_brain_mask(self, itk_img, th=70):
        img = sitk.GetArrayFromImage(itk_img)
        binary_img = self.binarize_img(img, itk_img, th=th)
        largest_cc_img_mask = self._get_largest_connected_component(binary_img)
        img_lcc_arr = sitk.GetArrayFromImage(largest_cc_img_mask)
        img_lcc_arr_filled = np.zeros(img_lcc_arr.shape).astype(np.uint8)
        for i, img_slice in enumerate(img_lcc_arr):
            img_lcc_arr_filled[i] = self._fill_hole(img_slice[:, :, None].astype(np.uint8))
        res_img = self.array2image(img_lcc_arr_filled, largest_cc_img_mask)
        return res_img
