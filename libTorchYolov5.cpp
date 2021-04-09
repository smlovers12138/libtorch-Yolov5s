#include"libTorchYolov5.h"

libTorchYolov5::libTorchYolov5()
{

}
libTorchYolov5::~libTorchYolov5()
{

}

int libTorchYolov5::torch_init(DeviceType u_device, string modelpath, string classFilepath, vector<string>& class_names, jit::script::Module& netModel)
{
	//º”‘ÿ¿‡±
	std::ifstream class_names_file(classFilepath.c_str());
	if (class_names_file.is_open())
	{
		std::string name = "";
		if (!class_names.empty()) class_names.clear();
		while (std::getline(class_names_file, name))
		{
			class_names.push_back(name);
		}
	}
	else
	{
		return -1;
	}
	if (modelpath.empty())
	{
		return -2;
	}
	try
	{
		netModel = jit::load(modelpath);
		netModel.to(u_device);
        //netModel.to(torch::kHalf);
        netModel.eval();
		Mat temp_img = cv::Mat::zeros(640, 640, CV_32FC3);
		temp_img.convertTo(temp_img, CV_32FC3, 1.0f / 255.0f);
		auto imgTensor = torch::from_blob(temp_img.data, { 1, temp_img.rows, temp_img.cols,temp_img.channels() }).to(u_device);
		imgTensor = imgTensor.permute({ 0, 3, 1, 2 }).contiguous(); 
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(imgTensor);
		torch::jit::IValue output = netModel.forward(inputs);
		auto preds = output.toTuple()->elements()[0].toTensor();
	}
	catch (...)
	{
		return -3;
	}


	return 0;
}

int libTorchYolov5::torch_detect(DeviceType u_device,jit::script::Module& netModel, Mat& InputImage, Size& InputSize, vector<Rect>& bboxes, vector<float>& confidence, vector<int>& classId)
{
	Mat Image, blob_image;
    //Image = InputImage.clone();
	//int image_height = Image.rows;
	//int image_width = Image.cols;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> pad_info = LetterboxImage(InputImage, blob_image, cv::Size(640, 640));
    const int pad_w = pad_info[0];
    const int pad_h = pad_info[1];
    const float scale = pad_info[2];

    cv::cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
    blob_image.convertTo(blob_image, CV_32FC3, 1.0f / 255.0f);
    
    auto imgTensor = torch::from_blob(blob_image.data, { 1, blob_image.rows, blob_image.cols,blob_image.channels() }).to(u_device);
    imgTensor = imgTensor.permute({ 0, 3, 1, 2 }).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width) 
    auto end = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "Pre takes : " << total_pre << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(imgTensor);
    torch::jit::IValue output = netModel.forward(inputs);
    auto preds = output.toTuple()->elements()[0].toTensor();
    end = std::chrono::high_resolution_clock::now();
    total_pre = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << "Forward takes : " << total_pre << " ms" << std::endl;

    //std::vector<std::vector<Detection>>Result_D;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Detection>>Result_D = PostProcessing(preds, pad_w, pad_h, scale, 0.5, 0.5);
    end = std::chrono::high_resolution_clock::now();
    total_pre = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "PostProcessing takes : " << total_pre << " ms" << std::endl;
    if (!Result_D.empty()) 
    {
        for (const auto& detection : Result_D[0]) 
        {
            const auto& box = detection.b_box;
            float score = detection.score;
            int class_idx = detection.class_idx;

            bboxes.push_back(box);
            confidence.push_back(score);
            classId.push_back(class_idx);
        }
    }

	return 0;
}


std::vector<float>libTorchYolov5:: LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = out_size.height;
	float out_w = out_size.width;

	float scale = std::min(out_w / in_w, out_h / in_h);

	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);

	cv::resize(src, dst, cv::Size(mid_w, mid_h));

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
	return pad_info;
}


torch::Tensor libTorchYolov5::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::zeros_like(x);
    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
    y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
    y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
    y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
    return y;
}


void libTorchYolov5:: Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
    const at::TensorAccessor<float, 2>& det,
    std::vector<cv::Rect>& offset_box_vec,
    std::vector<float>& score_vec)
{

    for (int i = 0; i < offset_boxes.size(0); i++) {
        offset_box_vec.emplace_back(
            cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
        );
        score_vec.emplace_back(det[i][Det::score]);
    }
}

std::vector<std::vector<Detection>> libTorchYolov5:: PostProcessing(const torch::Tensor& detections,
    int pad_w, int pad_h, float scale, const /*cv::Size& img_shape,*/
    float conf_thres, float iou_thres)
{
    constexpr int item_attr_size = 5;
    int batch_size = detections.size(0);
    auto num_classes = detections.size(2) - item_attr_size;

    auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

    std::vector<std::vector<Detection>> output;
    output.reserve(batch_size);

    
    // iterating all images in the batch
    for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
       
        auto start = std::chrono::high_resolution_clock::now();
        
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({ -1, num_classes + item_attr_size });

        auto end = std::chrono::high_resolution_clock::now();
        float total_pre = std::chrono::duration<float, std::milli>(end - start).count();
        std::cout << "masked_select takes : " << total_pre << " ms" << std::endl;
        // if none detections remain then skip and start to process next image
        if (0 == det.size(0)) {
            continue;
        }
       
        //compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
        det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);


        // box (center x, center y, width, height) to (x1, y1, x2, y2)
        torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

        // [best class only] get the max classes score at each result (e.g. elements 5-84)
        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

        // class score
        auto max_conf_score = std::get<0>(max_classes);
        // index
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

        // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
        det = torch::cat({ box.slice(1, 0, 4), max_conf_score, max_conf_index }, 1);

        // for batched NMS
        constexpr int max_wh = 4096;
        auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
        auto offset_box = det.slice(1, 0, 4) + c;

        std::vector<cv::Rect> offset_box_vec;
        std::vector<float> score_vec;



        // copy data back to cpu
        auto offset_boxes_cpu = offset_box.cpu();
        auto det_cpu = det.cpu();
        const auto& det_cpu_array = det_cpu.accessor<float, 2>();

        
        // use accessor to access tensor elements efficiently
        Tensor2Detection(offset_boxes_cpu.accessor<float, 2>(), det_cpu_array, offset_box_vec, score_vec);
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);
        
        std::vector<Detection> det_vec;
        for (int index : nms_indices)
        {
            Detection t;
            const auto& b = det_cpu_array[index];
            t.b_box =
                cv::Rect(cv::Point((b[Det::tl_x] - pad_w) / scale, (b[Det::tl_y] - pad_h) / scale),
                    cv::Point((b[Det::br_x] - pad_w) / scale, (b[Det::br_y] - pad_h) / scale));
            t.score = det_cpu_array[index][Det::score];
            t.class_idx = det_cpu_array[index][Det::class_idx];
            det_vec.emplace_back(t);
        }      
       
        output.emplace_back(det_vec);
       
    }
    
    return output;
}