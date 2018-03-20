#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <rcdraw.h>



using namespace cv;

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QTimer timer;

    VideoCapture *cap;
    RCDraw *visorS, *visorD;
    QImage *imgS, *imgD;
    Mat colorImage, grayImage, destColorImage, destGrayImage;
    Mat gray2ColorImage, destGray2ColorImage;
    bool capture, showColorImage, winSelected;
    Rect imageWindow;

    BFMatcher matcherObj;
    std::vector<std::vector<Mat>> arrayDescriptor1;
    std::vector<std::vector<Mat>> arrayDescriptor2;
    std::vector<std::vector<Mat>> arrayDescriptor3;
    Ptr<ORB> detector;

public slots:
    void compute();
    void start_stop_capture(bool start);
    void change_color_gray(bool color);
    void selectWindow(QPointF p, int w, int h);
    void deselectWindow();

    void extract_descriptor();
    void del_image();
    void update_matcher();
    void update_destImage();
    void detect_image();
    void drawObject(std::vector<Point2f> p1, std::vector<Point2f> p2, std::vector<Point2f> p3);

};


#endif // MAINWINDOW_H
