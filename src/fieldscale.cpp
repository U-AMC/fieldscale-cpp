/* Libs are to be updated */

using namespace cv;
using namespace std;

class Fieldscale {
public:
  Eigen::MatrixXd gridwiseMin(const cv::Mat& image, const cv::Size& gridShape) {
      int patchHeight = image.rows / gridShape.height;
      int patchWidth = image.cols / gridShape.width;

      Eigen::MatrixXd output(gridShape.height, gridShape.width);
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < gridShape.height; ++i) {
          for (int j = 0; j < gridShape.width; ++j) {
              cv::Rect roi(j * patchWidth, i * patchHeight, patchWidth, patchHeight);
              cv::Mat patch = image(roi);
              double minVal;
              cv::minMaxLoc(patch, &minVal, nullptr);
              output(i, j) = minVal;
          }
      }
      return output;
  }

  Eigen::MatrixXd gridwiseMax(const cv::Mat& image, const cv::Size& gridShape) {
      int patchHeight = image.rows / gridShape.height;
      int patchWidth = image.cols / gridShape.width;

      Eigen::MatrixXd output(gridShape.height, gridShape.width);
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < gridShape.height; ++i) {
          for (int j = 0; j < gridShape.width; ++j) {
              cv::Rect roi(j * patchWidth, i * patchHeight, patchWidth, patchHeight);
              cv::Mat patch = image(roi);
              double maxVal;
              cv::minMaxLoc(patch, nullptr, &maxVal);
              output(i, j) = maxVal;
          }
      }
      return output;
  }

  std::vector<std::pair<int, int>> getNeighborGrids(
      const Eigen::MatrixXd& grid, 
      const std::pair<int, int>& xy, 
      int maxDistance = 1) 
  {
      int h = grid.rows();
      int w = grid.cols();
      int x = xy.first;
      int y = xy.second;

      std::vector<std::pair<int, int>> neighbors;
      neighbors.reserve((2 * maxDistance + 1) * (2 * maxDistance + 1) - 1);

      int xStart = std::max(0, x - maxDistance);
      int xEnd = std::min(h - 1, x + maxDistance);
      int yStart = std::max(0, y - maxDistance);
      int yEnd = std::min(w - 1, y + maxDistance);

      for (int i = xStart; i <= xEnd; ++i) {
          for (int j = yStart; j <= yEnd; ++j) {
              if (i != x || j != y) {
                  neighbors.emplace_back(i, j);
              }
          }
      }

      return neighbors;
  }

Eigen::MatrixXd localExtremaSuppression(Eigen::MatrixXd grid, int localDistance, double diffThreshold, const std::string& extrema) {
    if (extrema != "max" && extrema != "min") {
        throw std::invalid_argument("Extrema must be 'max' or 'min'.");
    }
    if (localDistance <= 0 || diffThreshold <= 0) {
        return grid;
    }
    Eigen::MatrixXd result = grid;
    #pragma omp parallel for
    for (int i = 0; i < grid.rows(); ++i) {
        for (int j = 0; j < grid.cols(); ++j) {
            std::vector<std::pair<int, int>> neighbors = getNeighborGrids(grid, {i, j}, localDistance);
            std::vector<double> neighborValues;
            for (const auto& xy : neighbors) {
                neighborValues.push_back(grid(xy.first, xy.second));
            }
            double neighborMean = std::accumulate(neighborValues.begin(), neighborValues.end(), 0.0) / neighborValues.size();
            if (extrema == "max" && grid(i, j) >= *std::max_element(neighborValues.begin(), neighborValues.end())) {
                double diff = grid(i, j) - neighborMean;
                if (diff > diffThreshold) {
                    result(i, j) = neighborMean + diffThreshold;
                }
            } else if (extrema == "min" && grid(i, j) <= *std::min_element(neighborValues.begin(), neighborValues.end())) {
                double diff = neighborMean - grid(i, j);
                if (diff > diffThreshold) {
                    result(i, j) = neighborMean - diffThreshold;
                }
            }
        }
    }
    return result;
}

  Eigen::MatrixXd messagePassing(const Eigen::MatrixXd& grid, const std::string& direction) {
      if (direction != "increase" && direction != "decrease") {
          throw std::invalid_argument("Direction must be 'increase' or 'decrease'.");
      }
      Eigen::MatrixXd gridNew = Eigen::MatrixXd::Zero(grid.rows(), grid.cols());
      for (int i = 0; i < grid.rows(); ++i) {
          for (int j = 0; j < grid.cols(); ++j) {
              std::vector<std::pair<int, int>> neighbors = getNeighborGrids(grid, {i, j}, 1);
              std::vector<double> neighborValues;
              for (const auto& xy : neighbors) {
                  neighborValues.push_back(grid(xy.first, xy.second));
              }
              double mean = std::accumulate(neighborValues.begin(), neighborValues.end(), grid(i, j)) / (neighborValues.size() + 1);
              double bigger = std::max(mean, grid(i, j));
              double smaller = std::min(mean, grid(i, j));
              gridNew(i, j) = (direction == "increase") ? bigger : smaller;
          }
      }
      return gridNew;
  }

cv::Mat rescaleImageWithFields(const cv::Mat& image, const Eigen::MatrixXd& minField, const Eigen::MatrixXd& maxField) {
    cv::Mat rescaledImage = image.clone();
    rescaledImage.convertTo(rescaledImage, CV_64F);

    #pragma omp parallel for
    for (int i = 0; i < rescaledImage.rows; ++i) {
        for (int j = 0; j < rescaledImage.cols; ++j) {
            double pixelValue = rescaledImage.at<double>(i, j);
            double minVal = minField(i, j);
            double maxVal = maxField(i, j);

            if (minVal > maxVal) std::swap(minVal, maxVal);
            pixelValue = std::max(minVal, std::min(pixelValue, maxVal));
            rescaledImage.at<double>(i, j) = (pixelValue - minVal) / (maxVal - minVal) * 255.0;
        }
    }
    rescaledImage.convertTo(rescaledImage, CV_8U);
    return rescaledImage;
}


  Fieldscale(double maxDiff = 400.0, double minDiff = 400.0, int iteration = 30, double gamma = 2.0, bool clahe = true, bool video = false)
        : maxDiff(maxDiff), minDiff(minDiff), iteration(iteration), gamma(gamma), clahe(clahe), video(video), prevMinField(), prevMaxField() {
        if (maxDiff < 0 || minDiff < 0 || iteration <= 0 || gamma <= 0) {
            throw std::invalid_argument("Invalid parameter values.");
        }
    }

    cv::Mat operator()(const cv::Mat& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        cv::Mat image = input.clone();
        image.convertTo(image, CV_64F);

        Eigen::MatrixXd minGrid = gridwiseMin(image, cv::Size(8, 8));
        Eigen::MatrixXd maxGrid = gridwiseMax(image, cv::Size(8, 8));

        maxGrid = localExtremaSuppression(maxGrid, 2, maxDiff, "max");
        maxGrid = localExtremaSuppression(maxGrid, 2, minDiff, "min");

        for (int i = 0; i < iteration; ++i) {
            minGrid = messagePassing(minGrid, "decrease");
            maxGrid = messagePassing(maxGrid, "increase");
        }

        cv::Mat minField, maxField;
        cv::resize(eigenToCv(minGrid), minField, input.size(), 0, 0, cv::INTER_LINEAR);
        cv::resize(eigenToCv(maxGrid), maxField, input.size(), 0, 0, cv::INTER_LINEAR);

        if (video && !prevMinField.empty()) {
            minField = 0.1 * minField + 0.9 * prevMinField;
            maxField = 0.1 * maxField + 0.9 * prevMaxField;
        }

        prevMinField = minField;
        prevMaxField = maxField;

        cv::Mat rescaled = rescaleImageWithFields(input, cvToEigen(minField), cvToEigen(maxField));

        if (gamma > 0) {
            rescaled.convertTo(rescaled, CV_64F);
            cv::pow(rescaled / 255.0, gamma, rescaled);
            rescaled.convertTo(rescaled, CV_8U, 255);
        }

        if (clahe) {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(3, 3));
            clahe->apply(rescaled, rescaled);
        }

        return rescaled;
    }

cv::Mat eigenToCv(const Eigen::MatrixXd& eigenMatrix) {
    cv::Mat mat(eigenMatrix.rows(), eigenMatrix.cols(), CV_64F);
    for (int i = 0; i < eigenMatrix.rows(); ++i) {
        for (int j = 0; j < eigenMatrix.cols(); ++j) {
            mat.at<double>(i, j) = eigenMatrix(i, j);
        }
    }
    return mat;
}

Eigen::MatrixXd cvToEigen(const cv::Mat& cvMatrix) {
    Eigen::MatrixXd eigenMatrix(cvMatrix.rows, cvMatrix.cols);
    for (int i = 0; i < cvMatrix.rows; ++i) {
        for (int j = 0; j < cvMatrix.cols; ++j) {
            eigenMatrix(i, j) = cvMatrix.at<double>(i, j);
        }
    }
    return eigenMatrix;
}

private:
    double maxDiff;
    double minDiff;
    int iteration;
    double gamma;
    bool clahe;
    bool video;
    cv::Mat prevMinField;
    cv::Mat prevMaxField;

    // Eigen::MatrixXd gridwiseMin(const cv::Mat& image, const cv::Size& gridShape);
    // Eigen::MatrixXd gridwiseMax(const cv::Mat& image, const cv::Size& gridShape);
    // Eigen::MatrixXd localExtremaSuppression(Eigen::MatrixXd grid, int localDistance, double diffThreshold, const std::string& extrema);
    // Eigen::MatrixXd messagePassing(const Eigen::MatrixXd& grid, const std::string& direction);
    // cv::Mat rescaleImageWithFields(const cv::Mat& image, const Eigen::MatrixXd& minField, const Eigen::MatrixXd& maxField);
    // cv::Mat eigenToCv(const Eigen::MatrixXd& eigenMatrix);
    // Eigen::MatrixXd cvToEigen(const cv::Mat& cvMatrix);
};
