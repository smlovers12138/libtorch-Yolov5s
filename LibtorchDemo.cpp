#include"libTorchYolov5.h"

int main()
{
    libTorchYolov5 *yolov5detector = new libTorchYolov5();
    torch::DeviceType device_type = at::kCPU;
    if (torch::cuda::is_available())
        device_type = at::kCUDA;
    using torch::jit::script::Module;
    Module net;
    string weights_path = "Model/best.torchscriptG.pt";
    string class_name_path = "Model/Number.names";
    std::vector<string>class_names;
    yolov5detector->torch_init(device_type, weights_path, class_name_path, class_names, net);
    std::cout << "Load model successful!" << std::endl;
    while (1)
    {
        std::vector<float> confidences;
        std::vector<Rect> boxes;
        std::vector<int> classIds;
        cv::Mat Image, blob_image;
        cv::Size InputSize(640, 640);
        int input_imageh, input_imagew, inputn;
        Image = cv::imread("Test.bmp");
        auto start = std::chrono::high_resolution_clock::now();
        yolov5detector->torch_detect(device_type, net, Image,InputSize, boxes, confidences, classIds);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "takes : " << duration.count() << " ms" << std::endl;

        if (!boxes.empty()) {
            for (int i = 0; i < boxes.size(); i++)
            {

                cv::rectangle(Image, boxes[i], cv::Scalar(255, 0, 255), 2);

                if (1) 
                {
                    std::string s = class_names[classIds[i]];
                    auto font_face = cv::FONT_HERSHEY_DUPLEX;
                    auto font_scale = 1.0;
                    int thickness = 1;
                    int baseline = 0;
                    auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                    cv::rectangle(Image,
                        cv::Point(boxes[i].tl().x, boxes[i].tl().y - s_size.height - 5),
                        cv::Point(boxes[i].tl().x + s_size.width, boxes[i].tl().y),
                        cv::Scalar(0, 0, 255), -1);
                    cv::putText(Image, s, cv::Point(boxes[i].tl().x, boxes[i].tl().y - 5),
                        font_face, font_scale, cv::Scalar(255, 255, 0), thickness);
                }
            }
        }
        //if (dets.size() > 0)
        //{
        //    // Visualize result
        //    for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
        //    {
        //        float left = dets[0][i][0].item().toFloat() * Image.cols / 640;
        //        float top = dets[0][i][1].item().toFloat() * Image.rows / 640;
        //        float right = dets[0][i][2].item().toFloat() * Image.cols / 640;
        //        float bottom = dets[0][i][3].item().toFloat() * Image.rows / 640;
        //        float score = dets[0][i][4].item().toFloat();
        //        int classID = dets[0][i][5].item().toInt();
        //        std::stringstream ss2;
        //        ss2 << classID;
        //        cv::rectangle(Image, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);

        //        cv::putText(Image, ss2.str(),
        //            cv::Point(left, top),
        //            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        //    }
        //}
        // cv::putText(frame, "FPS: " + std::to_string(int(1e7 / (clock() - start))),
        //     cv::Point(50, 50),
        //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        std::cout << "\n" << std::endl;
        cv::namedWindow("",0);
        cv::imshow("", Image);
        if (cv::waitKey(100) == 27)
            break;
    }   

    return 0;
}
