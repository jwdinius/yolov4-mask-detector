/** @file utilities.hpp
 *  @author Joe Dinius, Ph.D
 *  @date 28 Nov. 2020
 *
 *  @brief helper function declarations for mask detector application using OpenCV and Yolov4 (Darknet)
 *
 *  @see https://docs.opencv.org/3.4/d4/db9/samples_2dnn_2object_detection_8cpp-example.html#_a20
 */
#pragma once
//! c/c++ headers
#include <vector>
//! system headers
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
//! project headers

/**
 * set up input to Yolo network
 */
void PreProcess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size const& inpSize) noexcept;

/**
 * get Yolo layer names
 */
std::vector<std::string> GetOutputsNames(cv::dnn::Net const& net) noexcept;

/**
 * post-process output from Yolov4
 */
void PostProcess(cv::Mat& frame, std::vector<cv::Mat> const& outs, std::vector<std::string> const& classes,
    double const& confThreshold, double const& nmsThreshold) noexcept;

/**
 * draw bounding box with identified class and confidence score on raw frame
 */
void DrawPred(std::string const& className, float const& conf, int const& left, int const& top, int const& right,
    int const& bottom, cv::Mat& frame) noexcept;
