1 caffe installation
  cd 01.caffe1.0 root
  make all -j 8
  make py

2 yolov5 training & onnx export
  cd 02.yolov5
  python train.py --cfg path/to/your/model.yaml # change nn.Upsample to nn.ConvTranspose2d
                  --data path/to/your/data.yaml
                  --noautoanchor
                  
  after training, using script bellow to detect imgs
  python detect.py --weights path/to/the/trained/model.pth
                   --source path/to/img/directory  

  export to onnx model
  python export.py --weights path/to/the/trained/model.pth
                   --train
                   --simplify
  
  check onnx output and pytorch output
  python onnx_forward.py path/to/onnx/model.onnx path/to/images # change the model parameters accordingly
  
3 convert to caffe model
  cd 03.onnx2caffe
  change caffe path settings in convertCaffe.py
  python convertCaffe.py path/to/onnx/model.onnx path/to/saved/model.prototxt path/to/saved/model.caffemodel
  
4 detect with caffe
  cd 01.caffe1.0
  change parameter settings in CAFFE_ROOT/tools/caffe_yolov5s.cpp
  make
  CAFFE_ROOT/build/tools/caffe_yolov5s path/to/caffe/model.prototxt path/to/caffe/model.caffemodel path/to/images
  
5 change code in hisi3559 on board official code
  first change the official yolov3 code to yolov5 code

  then change the yolov3 model parameter settings to yolov5 model parameter settings
  
  make
  enjoy!
  
