/** @file utilities.cpp
 *  @author Joe Dinius, Ph.D
 *  @date 28 Nov. 2020
 *
 *  @brief helper function definitions for mask detector application using OpenCV and Yolov4 (Darknet)
 *
 *  @see https://docs.opencv.org/3.4/d4/db9/samples_2dnn_2object_detection_8cpp-example.html#_a20
 */
//! c/c++ headers
#include <algorithm>
//! system headers
//! project headers
#include "utilities.hpp"

/**
 * set up input to Yolo network
 */
void PreProcess(cv::Mat const& frame, cv::dnn::Net& net, cv::Size const& inpSize) noexcept {
  if (inpSize.width <= 0 || inpSize.height <= 0) {
    std::cerr << "Incorrect input dimensions.  Exiting..." << std::endl;
    return;
  }

  //! Create a 4D blob from a frame.
  cv::Mat blob;
  cv::dnn::blobFromImage(frame, blob, 1. / 255., inpSize, cv::Scalar(), true, false);
  
  net.setInput(blob);
  return;
}

/**
 * get Yolo layer names
 * @note the internals of this method will only be executed once, to populate
 * static "names"
 */
std::vector<std::string> GetOutputsNames(cv::dnn::Net const& net) noexcept {
  static std::vector<std::string> names;
  if (names.empty()) {
    //! Get the indices of the output layers, i.e. the layers with unconnected outputs
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    
    //! get the names of all the layers in the network
    std::vector<std::string> layersNames = net.getLayerNames();
    
    //! Get the names of the output layers in names
    names.resize(outLayers.size());
    
    for (size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
    }
  }
  return names;
}

/**
 * post-process output from Yolov4
 */
void PostProcess(cv::Mat& frame, std::vector<cv::Mat> const& outs, std::vector<std::string> const& classes,
    double const& confThreshold, double const& nmsThreshold) noexcept {
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  
  for (size_t i = 0; i < outs.size(); ++i) {
    /** Scan through all the bounding boxes output from the network and keep only the
     * ones with high confidence scores. Assign the box's class label as the class
     * with the highest score for the box.*/
    float* data = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      cv::Point classIdPoint;
      double confidence;
      //! Get the value and location of the maximum score
      cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold) {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;
      
        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
  }
  
  /** Perform non maximum suppression to eliminate redundant overlapping boxes with
   * lower confidences */
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    cv::Rect box = boxes[idx];
    DrawPred(classes[classIds[idx]], confidences[idx], box.x, box.y,
        box.x + box.width, box.y + box.height, frame);
  }
}

/**
 * draw bounding box with identified class and confidence score on raw frame
 */
void DrawPred(std::string const& className, float const& conf, int const& left, int const& top,
    int const& right, int const& bottom, cv::Mat& frame) noexcept {
  //! Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));

  //! Get the label for the class name and its confidence
  std::string label = className + ":" + cv::format("%.2f", conf);

  //! Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  auto topClipped = std::max(top, labelSize.height);
  cv::putText(frame, label, cv::Point(left, topClipped), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
}
