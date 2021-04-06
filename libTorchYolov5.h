#pragma once
#include "data.h"
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;
using namespace cv;
using namespace torch;

class libTorchYolov5
{
public:
	libTorchYolov5();
	~libTorchYolov5();

	int torch_init(DeviceType u_device,string modelpath,string classFilepath, vector<string>& class_names, jit::script::Module &netModel);

	int torch_detect(DeviceType u_device, jit::script::Module& netModel, Mat& InputImage, Size& InputSize, vector<Rect>& bboxes, vector<float>& confidence, vector<int>& classId);
private:
	std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size);
	torch::Tensor xywh2xyxy(const torch::Tensor& x);
	void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes, const at::TensorAccessor<float, 2>& det, std::vector<cv::Rect>& offset_box_vec, std::vector<float>& score_vec);
	std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections, int pad_w, int pad_h, float scale, const float conf_thres, float iou_thres);
};
