#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    cap = new VideoCapture(1);
    if(!cap->isOpened())
        cap = new VideoCapture(1);
    capture = true;
    showColorImage = false;
    winSelected = false;
    cap->set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap->set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    imgS = new QImage(320,240, QImage::Format_RGB888);
    visorS = new RCDraw(320,240, imgS, ui->imageFrameS);
    imgD = new QImage(320,240, QImage::Format_RGB888);
    visorD = new RCDraw(320,240, imgD, ui->imageFrameD);

    colorImage.create(240,320,CV_8UC3);
    grayImage.create(240,320,CV_8UC1);
    destColorImage.create(240,320,CV_8UC3);
    destGrayImage.create(240,320,CV_8UC1);
    gray2ColorImage.create(240,320,CV_8UC3);
    destGray2ColorImage.create(240,320,CV_8UC3);

    matcherObj = BFMatcher(NORM_HAMMING);
    detector = ORB::create(500, 1.2, 8, 16);

    connect(&timer,SIGNAL(timeout()),this,SLOT(compute()));
    connect(ui->captureButton,SIGNAL(clicked(bool)),this,SLOT(start_stop_capture(bool)));
    connect(ui->colorButton,SIGNAL(clicked(bool)),this,SLOT(change_color_gray(bool)));
    connect(visorS,SIGNAL(windowSelected(QPointF, int, int)),this,SLOT(selectWindow(QPointF, int, int)));
    connect(visorS,SIGNAL(pressEvent()),this,SLOT(deselectWindow()));

    connect(ui->addButton, SIGNAL(clicked(bool)), this, SLOT(extract_descriptor()));
    connect(ui->delButton, SIGNAL(clicked(bool)), this, SLOT(del_image()));

    timer.start(60);


}

MainWindow::~MainWindow()
{
    delete ui;
    delete cap;
    delete visorS;
    delete visorD;
    delete imgS;
    delete imgD;

}

void MainWindow::compute()
{
    if(capture && cap->isOpened())
    {
        *cap >> colorImage;

        cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        cvtColor(colorImage, colorImage, CV_BGR2RGB);
    }

    update_destImage();
    detect_image();

    if(showColorImage)
    {
        memcpy(imgS->bits(), colorImage.data , 320*240*3*sizeof(uchar));
        memcpy(imgD->bits(), destColorImage.data , 320*240*3*sizeof(uchar));
    }
    else
    {
        cvtColor(grayImage,gray2ColorImage, CV_GRAY2RGB);
        cvtColor(destGrayImage,destGray2ColorImage, CV_GRAY2RGB);
        memcpy(imgS->bits(), gray2ColorImage.data , 320*240*3*sizeof(uchar));
        memcpy(imgD->bits(), destGray2ColorImage.data , 320*240*3*sizeof(uchar));
    }

    if(winSelected)
    {
        visorS->drawSquare(QPointF(imageWindow.x+imageWindow.width/2, imageWindow.y+imageWindow.height/2), imageWindow.width,imageWindow.height, Qt::green );
    }
    visorS->update();
    visorD->update();

}

void MainWindow::start_stop_capture(bool start)
{
    if(start)
    {
        ui->captureButton->setText("Stop capture");
        capture = true;
    }
    else
    {
        ui->captureButton->setText("Start capture");
        capture = false;
    }
}

void MainWindow::change_color_gray(bool color)
{
    if(color)
    {
        ui->colorButton->setText("Gray image");
        showColorImage = true;
    }
    else
    {
        ui->colorButton->setText("Color image");
        showColorImage = false;
    }
}

void MainWindow::selectWindow(QPointF p, int w, int h)
{
    QPointF pEnd;
    if(w>0 && h>0)
    {
        imageWindow.x = p.x()-w/2;
        if(imageWindow.x<0)
            imageWindow.x = 0;
        imageWindow.y = p.y()-h/2;
        if(imageWindow.y<0)
            imageWindow.y = 0;
        pEnd.setX(p.x()+w/2);
        if(pEnd.x()>=320)
            pEnd.setX(319);
        pEnd.setY(p.y()+h/2);
        if(pEnd.y()>=240)
            pEnd.setY(239);
        imageWindow.width = pEnd.x()-imageWindow.x;
        imageWindow.height = pEnd.y()-imageWindow.y;

        winSelected = true;
    }
}

void MainWindow::deselectWindow()
{
    winSelected = false;
}

void MainWindow::extract_descriptor(){

    std::vector<KeyPoint> keyPoints;
    Mat descriptors;

    Mat win;
    win.create(imageWindow.height,imageWindow.width,CV_8UC1);
    Mat(grayImage, imageWindow).copyTo(win);

    detector->detect(grayImage(imageWindow),keyPoints);
    detector->compute(grayImage(imageWindow), keyPoints, descriptors);
    //qDebug()<<keyPoints.size() << "img"<< imageWindow.height << imageWindow.width;

    /*for(int i=0; i<keyPoints.size();i++)
    {
        int iniX=0;//(320-imageWindow.width)/2;
        int iniY=0;//(240-imageWindow.height)/2;
        visorS->drawSquare(QPointF(iniX+keyPoints[i].pt.x,iniY+keyPoints[i].pt.y),5,5,Qt::red, true) ;
    }*/

    if(keyPoints.size()>4){
        std::vector<Mat> obj = {win,descriptors};
        switch (ui->comboBox->currentIndex()) {
        case 0:
            arrayDescriptor1.push_back(obj); //push_back añade al final    // para quitarlo pop_back(); creo
            ui->slider->setMaximum(arrayDescriptor1.size()-1);
            ui->slider->setSliderPosition(arrayDescriptor1.size());
            break;
        case 1:
            arrayDescriptor2.push_back(obj);
            ui->slider->setMaximum(arrayDescriptor2.size()-1);
            ui->slider->setSliderPosition(arrayDescriptor2.size());
            break;
        case 2:
            arrayDescriptor3.push_back(obj);
            ui->slider->setMaximum(arrayDescriptor3.size()-1);
            ui->slider->setSliderPosition(arrayDescriptor3.size());
            break;
        default:
            break;
        }

        update_matcher();
    }else{ qDebug()<<"Few characteristic points";}
}

void MainWindow::del_image(){
    switch (ui->comboBox->currentIndex()) {
    case 0:
        arrayDescriptor1.erase(arrayDescriptor1.begin()+ui->slider->value());
        ui->slider->setMaximum(arrayDescriptor1.size()-1);
        ui->slider->setSliderPosition(arrayDescriptor1.size());
        break;
    case 1:
        arrayDescriptor2.erase(arrayDescriptor2.begin()+ui->slider->value());
        ui->slider->setMaximum(arrayDescriptor2.size()-1);
        ui->slider->setSliderPosition(arrayDescriptor2.size());
        break;
    case 2:
        arrayDescriptor3.erase(arrayDescriptor3.begin()+ui->slider->value());
        ui->slider->setMaximum(arrayDescriptor3.size()-1);
        ui->slider->setSliderPosition(arrayDescriptor3.size());
        break;
    default:
        break;
    }

    update_matcher();
}

void MainWindow::update_matcher(){
    matcherObj.clear();
    for(auto a: arrayDescriptor1)
        matcherObj.add(a.at(1));

    for(auto a: arrayDescriptor2)
        matcherObj.add(a.at(1));

    for(auto a: arrayDescriptor3)
        matcherObj.add(a.at(1));
}

void MainWindow::update_destImage(){

    int x, y;
    Mat winD;
    try{
        destGrayImage.setTo(0);
        switch (ui->comboBox->currentIndex()) {
        case 0:
            if(arrayDescriptor1.size()>0){
                ui->slider->setMaximum(arrayDescriptor1.size()-1);
                x = (320-arrayDescriptor1.at(ui->slider->value()).at(0).cols)/2;
                y = (240-arrayDescriptor1.at(ui->slider->value()).at(0).rows)/2;

                winD = destGrayImage(cv::Rect(x, y, arrayDescriptor1.at(ui->slider->value()).at(0).cols, arrayDescriptor1.at(ui->slider->value()).at(0).rows));
                arrayDescriptor1.at(ui->slider->value()).at(0).copyTo(winD);
            }else{
                ui->slider->setSliderPosition(0);
                ui->slider->setMaximum(0);
            }
            break;
        case 1:
            if(arrayDescriptor2.size()>0){
                ui->slider->setMaximum(arrayDescriptor2.size()-1);
                x = (320-arrayDescriptor2.at(ui->slider->value()).at(0).cols)/2;
                y = (240-arrayDescriptor2.at(ui->slider->value()).at(0).rows)/2;

                winD = destGrayImage(cv::Rect(x, y, arrayDescriptor2.at(ui->slider->value()).at(0).cols, arrayDescriptor2.at(ui->slider->value()).at(0).rows));
                arrayDescriptor2.at(ui->slider->value()).at(0).copyTo(winD);
            }else{
                ui->slider->setSliderPosition(0);
                ui->slider->setMaximum(0);
            }
            break;
        case 2:
            if(arrayDescriptor3.size()>0){
                ui->slider->setMaximum(arrayDescriptor3.size()-1);
                x = (320-arrayDescriptor3.at(ui->slider->value()).at(0).cols)/2;
                y = (240-arrayDescriptor3.at(ui->slider->value()).at(0).rows)/2;

                winD = destGrayImage(cv::Rect(x, y, arrayDescriptor3.at(ui->slider->value()).at(0).cols, arrayDescriptor3.at(ui->slider->value()).at(0).rows));
                arrayDescriptor3.at(ui->slider->value()).at(0).copyTo(winD);
            }else{
                ui->slider->setSliderPosition(0);
                ui->slider->setMaximum(0);
            }
            break;
        default:
            break;
        }
    }catch(...){
        qDebug()<<"Error update_image";
    }
}


void MainWindow::detect_image(){

    std::vector<KeyPoint> keyPointsSrc;
    std::vector<Point2f> points1;
    std::vector<Point2f> points2;
    std::vector<Point2f> points3;

    Mat descriptors;
    Point2f pt;
    //Mat win;
    //win.create(240,320,CV_8UC1);
    //grayImage.copyTo(win);
    uint result = 0;
    uint img = 0;
    detector->detectAndCompute(grayImage, Mat(), keyPointsSrc, descriptors, false);
    std::vector<DMatch> matches; //correspondencias
    matcherObj.match(descriptors, matches, Mat());

    for(uint i = 0; i<matches.size(); i++){
        if(matches[i].distance < 50){
            pt = keyPointsSrc[matches[i].queryIdx].pt;
            img = matches[i].imgIdx;
            if(img < arrayDescriptor1.size())
                points1.push_back(pt);
            else{
                result = arrayDescriptor1.size() + arrayDescriptor2.size();
                if(img < result)
                    points2.push_back(pt);
                else
                    points3.push_back(pt);
            }
        }
    }

    drawObject(points1, points2, points3);
}

void MainWindow::drawObject(std::vector<Point2f> p1, std::vector<Point2f> p2, std::vector<Point2f> p3){
    if(p1.size()>4){
        int maxX = 0, maxY = 0, minX = 320, minY =240;
        for(uint i = 0; i<p1.size(); i++){
            if(p1[i].x>maxX)
                maxX = p1[i].x;
            if(p1[i].y>maxY)
                maxY = p1[i].y;
            if(p1[i].x<minX)
                minX = p1[i].x;
            if(p1[i].y<minY)
                minY = p1[i].y;
        }

        visorS->drawSquare(QPointF(minX,minY),maxX-minX,maxY-minY,Qt::red, false) ;

    }

    //if(p2.size()>4){

    //}

    //if(p3.size()>4){

    //}
}
