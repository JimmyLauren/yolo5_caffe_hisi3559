
1. caffe installation  
	cd 01.caffe1.0  
	change Makefile.config (the uploaded Makefile.config file is well configured.)  
	make all -j 8  
	make py  
	  
2. yolov5 model training(Anaconda in recommended)  
	cd 02.yolov5  
	modify model.yaml  
		change nn.Upsample to nn.ConvTranspose2d  
	modify data.yaml  
	training using script bellow  
		python train.py --weights path/to/pretrained/yolov5s.pt   
						--cfg path/to/model.yaml  
						--data path/to/data.yaml  
						--imgsz 416  
						--noautoanchor  
	detect images using script bellow  
		python detect.py --weights path/to/model.pt  
						 --source path/to/images  
						 --imgsz 416  
	convert to onnx model  
		python export.py --weights path/to/trained/model.pt  
						 --train  
						 --simplify  
						 --opset 10  
	check the uniformity of pytorch output and onnx output  
		python onnx_forward.py path/to/onnx/model.onnx path/to/images  
		
3. caffe model export & detect  
	cd 03.onnx2caffe  
	change path settings in convertCaffe.py  
	python convertCaffe.py path/to/onnx/model.onnx path/to/caffe/model.prototxt path/to/caffe/model.caffemodel  
	
	cd 01.caffe1.0  
	change CAFFE_ROOT/tools/caffe_yolov5s.cpp model settings  
	make   
	CAFFE_ROOT/build/tools/caffe_yolov5s path/to/caffe/model.prototxt path/to/caffe/model.caffemodel path/to/images  
	
4. hisi3559 on board code modification  
	first change yolov3 postprocess code to yolov5 postprocess code:  
  ![6053c5678c79158de81db257085da6f](https://user-images.githubusercontent.com/15936220/172119359-fea56e23-0cac-4f24-b956-cecfcb1da623.png)
![22a9616082f040fb5699cb57e987f28](https://user-images.githubusercontent.com/15936220/172119381-13f8159e-69ff-4a5c-aad2-ccfd47b2e4e9.png)
![ef393522edbb46652a2feb4e9067a55](https://user-images.githubusercontent.com/15936220/172119436-db790f17-f039-4003-8cd2-eb28deeade60.png)

	
	then change yolov3 model parameters settings to yolov5 model parameters settings:  
	change code in func SAMPLE_SVP_NNIE_Yolov3_SoftwareInit  
  ![ced2319708b164726379e46b8231293](https://user-images.githubusercontent.com/15936220/172119493-8f1395ff-6380-4496-ac8b-abd2d8151137.png)
