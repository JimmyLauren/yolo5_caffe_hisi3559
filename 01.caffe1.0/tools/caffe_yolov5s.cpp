#include "caffe/caffe.hpp"
#include <string>
#include <vector>
#include <sys/io.h>
#include <unistd.h>
#include <stdio.h>

#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

// 用于计时
#include <boost/date_time/posix_time/posix_time.hpp>

#define INPUT_W 640
#define INPUT_H 384
// #define IsPadding 0
#define NUM_CLASS 2
#define NMS_THRESH 0.45
#define CONF_THRESH 0.11
//std::string prototxt_path = "../model/yolov5s-4.0-focus.prototxt";
//std::string caffemodel_path = "../model/yolov5s-4.0-focus.caffemodel";
//std::string pic_path = "/home/willer/calibration_data/2ad80d25-b022-3b9d-a46f-853f112c2dfe.jpg";

using namespace cv;
using namespace std;
using namespace caffe;
using std::string;

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::string;
using caffe::vector;
using std::cout;
using std::endl;
using std::ostringstream;

struct Bbox{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int cid;
};

struct Anchor{
    float width;
    float height;
};



bool get_all_files(const std::string& dir_in, std::vector<std::string>& files) {
    if (dir_in.empty()) {
        return false;
    }
    struct stat s;
    stat(dir_in.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        return false;
    }
    DIR* open_dir = opendir(dir_in.c_str());
    if (NULL == open_dir) {
        std::exit(EXIT_FAILURE);
    }
    dirent* p = nullptr;
    while( (p = readdir(open_dir)) != nullptr) {
        struct stat st;
        if (p->d_name[0] != '.') {
            //因为是使用devC++ 获取windows下的文件，所以使用了 "\" ,linux下要换成"/"
            std::string name = dir_in + std::string("/") + std::string(p->d_name);
            stat(name.c_str(), &st);
            if (S_ISDIR(st.st_mode)) {
                get_all_files(name, files);
            }
            else if (S_ISREG(st.st_mode)) {
                files.push_back(name);
            }
        }
    }
    closedir(open_dir);
    return true;
}

std::vector<Anchor> initAnchors(){
    std::vector<Anchor> anchors;
    Anchor anchor;
    // 13,16, 26,29, 47,47, 69,68, 118,97, 102,163, 229,135, 190,208, 358,248
    anchor.width = 13;
    anchor.height = 16;
    anchors.emplace_back(anchor);
    anchor.width = 26;
    anchor.height = 29;
    anchors.emplace_back(anchor);
    anchor.width = 47;
    anchor.height = 47;
    anchors.emplace_back(anchor);
    anchor.width = 69;
    anchor.height = 68;
    anchors.emplace_back(anchor);
    anchor.width = 118;
    anchor.height = 97;
    anchors.emplace_back(anchor);
    anchor.width = 102;
    anchor.height = 163;
    anchors.emplace_back(anchor);
    anchor.width = 229;
    anchor.height = 135;
    anchors.emplace_back(anchor);
    anchor.width = 190;
    anchor.height = 208;
    anchors.emplace_back(anchor);
    anchor.width = 358;
    anchor.height = 248;
    anchors.emplace_back(anchor);
    return anchors;
}

template <typename T>
T clip(const T &n, const T &lower, const T &upper){
    return std::max(lower, std::min(n, upper));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

void yoloTransform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<Bbox> &bboxes
               /*bool is_padding*/) 
{
    for (auto &bbox : bboxes)
    {
        bbox.xmin = bbox.xmin * iw / ow;
        bbox.ymin = bbox.ymin * ih / oh;
        bbox.xmax = bbox.xmax * iw / ow;
        bbox.ymax = bbox.ymax * ih / oh;
    }
}


cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes){
    for (auto it: bboxes){
        float score = it.score;
        int id = it.cid;
        cv::rectangle(image, cv::Point(it.xmin, it.ymin), cv::Point(it.xmax, it.ymax), cv::Scalar(255, 0,0), 3);
        cv::putText(image, std::to_string(id)+"_"+std::to_string(score), cv::Point(it.xmin, it.ymin), cv::FONT_HERSHEY_COMPLEX, 2.0, cv::Scalar(0,0,255));
    }
    return image;
}

void nms_cpu(std::vector<Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    std::vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
    }
    // 3.循环
    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
            float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
            float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
            float width = std::max(right - left + 1, 0.f);
            float height = std::max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}
template <typename T>
T sigmoid(const T &n) {
    return 1 / (1 + exp(-n));
}
void postProcessParall(const int height, const int width, int scale_idx, float postThres, float * origin_output, vector<int> Strides, vector<Anchor> Anchors, vector<Bbox> *bboxes)
{
    Bbox bbox;
    float cx, cy, w_b, h_b, score;
    int cid;
    const float *ptr = (float *)origin_output;
    for(unsigned long a=0; a<3; ++a){
        for(unsigned long h=0; h<height; ++h){
            for(unsigned long w=0; w<width; ++w){
                const float *cls_ptr =  ptr + 5;
                cid = argmax(cls_ptr, cls_ptr+NUM_CLASS);
                score = sigmoid(ptr[4]) * sigmoid(cls_ptr[cid]);

                if(score>=postThres)
                {
                    cx = (sigmoid(ptr[0]) * 2.f - 0.5f + static_cast<float>(w)) * static_cast<float>(Strides[scale_idx]);
                    cy = (sigmoid(ptr[1]) * 2.f - 0.5f + static_cast<float>(h)) * static_cast<float>(Strides[scale_idx]);
                    w_b = powf(sigmoid(ptr[2]) * 2.f, 2) * Anchors[scale_idx * 3 + a].width;
                    h_b = powf(sigmoid(ptr[3]) * 2.f, 2) * Anchors[scale_idx * 3 + a].height;
                    bbox.xmin = clip(cx - w_b / 2, 0.f, static_cast<float>(INPUT_W - 1));
                    bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
                    bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(INPUT_W - 1));
                    bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
                    bbox.score = score;
                    bbox.cid = cid+1;
                    /*
                    std::cout<< "bbox.cid : " << bbox.cid << std::endl;
										std::cout<< "bbox.score : " << bbox.score << std::endl;
										std::cout<< "bbox.xmin : " << bbox.xmin << std::endl;
										std::cout<< "bbox.ymin : " << bbox.ymin << std::endl;
										std::cout<< "bbox.xmax : " << bbox.xmax << std::endl;
										std::cout<< "bbox.ymax : " << bbox.ymax << std::endl;
										*/
                    bboxes->push_back(bbox);
                }
                ptr += 5 + NUM_CLASS;
            }
        }
    }
}
vector<Bbox> postProcess(vector<float *> origin_output, float postThres, float nmsThres) {

    vector<Anchor> Anchors = initAnchors();
    vector<Bbox> bboxes;
    vector<int> Strides = vector<int> {8, 16, 32};
    for (int scale_idx=0; scale_idx<3; ++scale_idx) {
        const int stride = Strides[scale_idx];
        const int width = (INPUT_W + stride - 1) / stride;
        const int height = (INPUT_H + stride - 1) / stride;
        //std::cout << "width : " << width << " " << "height : " << height << std::endl;
        float * cur_output_tensor = origin_output[scale_idx];
        postProcessParall(height, width, scale_idx, postThres, cur_output_tensor, Strides, Anchors, &bboxes);
    }
    nms_cpu(bboxes, nmsThres);
    return bboxes;
}

cv::Mat preprocess_img(cv::Mat& img /*, bool is_padding*/) 
{
    cv::Mat out;
    cv::resize(img,out,cv::Size(INPUT_W,INPUT_H),cv::INTER_LINEAR);
    return out;
}

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cout<<"usage: exe prototxt caffemodel jpg_list_file"<<std::endl;
		return -1;
	}
	std::string prototxt_path = argv[1];
	std::string caffemodel_path = argv[2];
	std::string pic_path = argv[3];

    ::google::InitGoogleLogging("caffe"); //初始化日志文件,不调用会给出警告,但不会报错
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_rank(1); //不进行日志输出
    Net<float> caffe_net(prototxt_path, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(caffemodel_path);


	vector<string> files;
	get_all_files(pic_path,files);
	
	for(int idx=0; idx<files.size(); idx++)
	{
		// 读入图片
		cv::Mat img = cv::imread(files[idx]);
		//cv::Mat img = cv::imread("../model/21.jpg");
		CHECK(!img.empty()) << "Unable to decode image ";
		cv::Mat showImage = img.clone();

		// 图片预处理,并加载图片进入blob
		Blob<float> *input_layer = caffe_net.input_blobs()[0];
		float *input_data = input_layer->mutable_cpu_data();

		static float data[3 * INPUT_H * INPUT_W];
		cv::Mat pre_img = preprocess_img(img /*,IsPadding*/);
		std::cout << "preprocess_img finished!\n";
		int i = 0;
		for (int row = 0; row < INPUT_H; ++row) 
        {
			uchar* uc_pixel = pre_img.data + row * pre_img.step;
			for (int col = 0; col < INPUT_W; ++col) 
            {  // opencv读取原始格式为bgr格式，按照0,1,2顺序为bgr，2,1,0顺序为rgb格式
				data[i] = (float)uc_pixel[0] / 255.0;
				data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
				data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[2] / 255.0;
				uc_pixel += 3;
				++i;
			}
		}
		
#if 0
        printf("begin save txt.\n");
		FILE* fWrite = fopen("input.txt","w");
        for(int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        {
            fprintf(fWrite,"%.0f \n",data[i]*255.0);
        }
		fclose(fWrite);
        printf("end save txt.\n");
#endif
    
		memcpy((float *) (input_data),
			   data, 3 * INPUT_H * INPUT_W * sizeof(float));

		//boost::posix_time::ptime start_time_ = boost::posix_time::microsec_clock::local_time(); //开始计时
		float total_time = 0;    
		//前向运算
		int nums = 1;

        boost::posix_time::ptime start_time_1 = boost::posix_time::microsec_clock::local_time();
        caffe_net.Forward();
        boost::posix_time::ptime end_time_1 = boost::posix_time::microsec_clock::local_time();
        total_time += (end_time_1 - start_time_1).total_milliseconds(); 
        std::cout << "[ " << idx << " ] " << (end_time_1 - start_time_1).total_milliseconds() << " ms." << std::endl;


		//boost::posix_time::ptime end_time_ = boost::posix_time::microsec_clock::local_time(); //结束计时

		Blob<float> *output_layer0 = caffe_net.output_blobs()[2];
		const float *output0 = output_layer0->cpu_data();
		cout << "output0 shape: " << output_layer0->shape(0) << " " << output_layer0->shape(1) << " " <<  output_layer0->shape(2) << " " <<  output_layer0->shape(3) <<  " " <<  output_layer0->shape(4) << endl;

		Blob<float> *output_layer1 = caffe_net.output_blobs()[0];
		const float *output1 = output_layer1->cpu_data();
		cout << "output1 shape: " << output_layer1->shape(0) << " " << output_layer1->shape(1) << " " <<  output_layer1->shape(2) << " " <<  output_layer1->shape(3) <<  " " <<  output_layer1->shape(4) << endl;

		Blob<float> *output_layer2 = caffe_net.output_blobs()[1];
		const float *output2 = output_layer2->cpu_data();
		cout << "output2 shape: " << output_layer2->shape(0) << " " << output_layer2->shape(1) << " " <<  output_layer2->shape(2) << " " <<  output_layer2->shape(3) <<  " " <<  output_layer2->shape(4) << endl;

		vector<float *> cur_output_tensors;
		cur_output_tensors.push_back(const_cast<float *>(output0));
		cur_output_tensors.push_back(const_cast<float *>(output1));
		cur_output_tensors.push_back(const_cast<float *>(output2));

		vector<Bbox> bboxes = postProcess(cur_output_tensors, CONF_THRESH, NMS_THRESH);
		

		yoloTransform(showImage.rows, showImage.cols, INPUT_H, INPUT_W, bboxes /*, IsPadding*/);
		showImage = renderBoundingBox(showImage, bboxes);

		string sFileName = files[idx].substr(files[idx].find_last_of("/")+1);
		sFileName = "img_result/"+sFileName;
		cv::imwrite(sFileName.c_str(), showImage);


		std::cout << "average time : " << total_time / nums*1.0 << " ms" << std::endl;
	}
}
